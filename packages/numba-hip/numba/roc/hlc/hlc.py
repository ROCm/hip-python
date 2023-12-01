# A temporary wrapper to connect to the HLC LLVM binaries.
# Currently, connect to commandline interface.
import sys
from subprocess import check_call, check_output
import subprocess
import tempfile
import os
import re
from collections import namedtuple
from numba.roc.hsadrv import devices
from .common import AMDGCNModule
from .config import ROCM_BC_PATH, OCLC_ABI_VERSION
from numba.roc.hlc import TRIPLE
from datetime import datetime
from contextlib import contextmanager
from numba.core import utils, config
from numba.roc.hsadrv.error import HsaSupportError

_real_check_call = check_call

NOISY_CMDLINE = True

@contextmanager
def error_pipe():
    if NOISY_CMDLINE:
       yield subprocess.STDOUT
    else:
        yield subprocess.DEVNULL


def check_call(caller,*args, **kwargs):
    # This is so that time is stamped against invocation
    # such that correlations can be looked for against messages in the
    # sys and kernel logs.
    try:
        with error_pipe() as stderr:
            if NOISY_CMDLINE:
                print(datetime.now().strftime("%b %d %H:%M:%S"),
                      file=sys.stdout)
                print(f'caller: {caller} - CMD: ' + ';'.join(args), file=sys.stdout)

            ret = _real_check_call(*args, stderr=stderr, **kwargs)

    except subprocess.CalledProcessError as e:
        print(e)
        raise(e)
    return ret


class CmdLine(object):

    def _initialize(self):
        if not self.initialized:
            dev_ctx = devices.get_context()
            self.arch = dev_ctx.agent.name.decode('UTF-8')
            self.target_cpu = "--mcpu %s" % self.arch

        self.CMD_OPT = ' '.join([
                self.opt,
                "-passes='default<O3>,verify'",
                self.triple_flag,
                self.target_cpu,
                "--disable-simplify-libcalls",
                "-S",
                "-o {fout}",
                "{fin}"])

        self.CMD_VERIFY = ' '.join([
                    self.opt,
                    "-passes='verify'",
                    self.triple_flag,
                    self.target_cpu,
                    "-S",
                    "-o {fout}",
                    "{fin}"])

        self.CMD_GEN_HSAIL = ' '.join([self.llc,
                        "-O2",
                        self.triple_flag,
                        self.target_cpu,
                        "--filetype=asm",
                        "-o {fout}",
                        "{fin}"])

        self.CMD_GEN_BRIG = ' '.join([self.llc,
                        "-O2",
                        self.triple_flag,
                        self.target_cpu,
                        "--filetype=obj",
                        "-o {fout}",
                        "{fin}"])

        self.CMD_LINK_BUILTINS = ' '.join([
                            self.llvm_link,
                            "-S",
                            "-o {fout}",
                            "{fin}",
                            "{lib}"])

        self.CMD_LINK_LIBS = ' '.join([self.llvm_link,
                        "-S",
                        "-o {fout}",
                        "{fin}"])

        self.CMD_LINK_BRIG = ' '.join([self.ld_lld,
                        "--shared",
                        "-o {fout}",
                        "{fin}"])

    def __init__(self):
        self._binary_path = os.environ.get('HSAILBIN', None)
        def _setup_path(tool):
            if self._binary_path is not None:
                return os.path.join(self._binary_path, tool)
            else:
                binpath = os.path.join(sys.prefix, 'bin', tool)
                return binpath
        self._triple = TRIPLE

        self.opt = _setup_path("opt")
        self.llc = _setup_path("llc")
        self.llvm_link = _setup_path("llvm-link")
        self.ld_lld = _setup_path("ld.lld")
        self.triple_flag = "--mtriple %s" % self._triple
        self.initialized = False

    def check_tooling(self):
        # make sure the llc can actually target amdgcn, ideally all tooling
        # should be checked but most don't print anything useful and so
        # compilation for AMDGCN would have to be tested instead. This is a
        # smoke test like check.
        try:
            if not os.path.isfile(self.llc):
                raise HsaSupportError('llc not found')
            output = check_output([self.llc, '--version'],
                                  universal_newlines=True)
            olines = [x.strip() for x in output.splitlines()]
            tgtidx = olines.index('Registered Targets:')
            targets = olines[tgtidx + 1:]
            for tgt in targets:
                if 'amdgcn' in tgt:
                    break
            else:
                msg = 'Command line tooling does not support "amdgcn" target'
                raise HsaSupportError(msg)
        except Exception as e:
            raise

    def verify(self, ipath, opath):
        if not self.initialized:
            self._initialize()
        check_call("verify",self.CMD_VERIFY.format(fout=opath, fin=ipath), shell=True)

    def optimize(self, ipath, opath):
        if not self.initialized:
            self._initialize()
        check_call("optimize",self.CMD_OPT.format(fout=opath, fin=ipath), shell=True)

    def generate_hsail(self, ipath, opath):
        if not self.initialized:
            self._initialize()
        check_call("generate_hsail",self.CMD_GEN_HSAIL.format(fout=opath, fin=ipath), shell=True)

    def generate_brig(self, ipath, opath):
        if not self.initialized:
            self._initialize()
        check_call("generate_brig",self.CMD_GEN_BRIG.format(fout=opath, fin=ipath), shell=True)

    def link_libs(self, ipath, libpaths, opath):
        if not self.initialized:
            self._initialize()
        cmdline = self.CMD_LINK_LIBS.format(fout=opath, fin=ipath)
        cmdline += ' '.join(["{0}".format(lib) for lib in libpaths])
        check_call("link_libs",cmdline, shell=True)

    def link_brig(self, ipath, opath):
        if not self.initialized:
            self._initialize()
        check_call("link_brig",self.CMD_LINK_BRIG.format(fout=opath, fin=ipath), shell=True)


class Module(AMDGCNModule):
    def __init__(self):
        """
        Setup
        """
        self._tmpdir = tempfile.mkdtemp()
        self._tempfiles = []
        self._linkfiles = []
        self._cmd = CmdLine()
        AMDGCNModule.__init__(self)

    def __del__(self):
        return
        self.close()

    def close(self):
       # Remove all temporary files
        for afile in self._tempfiles:
            os.unlink(afile)
        #Remove directory
        os.rmdir(self._tmpdir)

    def _create_temp_file(self, name, mode='wb'):
        path = self._track_temp_file(name)
        fobj = open(path, mode=mode)
        return fobj, path

    def _track_temp_file(self, name):
        path = os.path.join(self._tmpdir,
                            "{0}-{1}".format(len(self._tempfiles), name))
        self._tempfiles.append(path)
        return path

    def load_llvm(self, llvmir):
        """
        Load LLVM with HSAIL SPIR spec
        """
        # Preprocess LLVM IR
        with open("1_llvmir_from_numba.ll","w") as outfile:
            outfile.write(llvmir)

        llvmir = self._preprocess(llvmir)

        with open("2_llvmir_preprocessed.ll","w") as outfile:
            outfile.write(llvmir)

        # Create temp file to store the input file
        tmp_llvm_ir, fin = self._create_temp_file("dump-llvm-ir")
        with tmp_llvm_ir:
            tmp_llvm_ir.write(llvmir.encode('ascii'))

        # Create temp file for optimization
        fout = self._track_temp_file("verified-llvm-ir")
        self._cmd.verify(ipath=fin, opath=fout)

        if config.DUMP_OPTIMIZED:
            with open(fout, 'rb') as fin_opt:
                print(fin_opt.read().decode('ascii'))

        self._linkfiles.append(fout)

    def link_builtins(self, ipath, opath):
        bc_files = [os.path.join(ROCM_BC_PATH, bc) for bc in self.iter_bitcodes(self._cmd.arch,abi_ver=OCLC_ABI_VERSION)]
        cmd = self._cmd.CMD_LINK_BUILTINS.format(fout=opath, fin=ipath, lib=" ".join(bc_files))
        check_call("link_builtins", cmd, shell=True)

    def generateGCN(self):
        """
        Generate GCN from a module and also return the HSAIL code.
        """
        assert not self._finalized, "Module already has GCN generated"

        # Link dependencies libraries
        llvmfile = self._linkfiles[0]
        pre_builtin_path = self._track_temp_file("link-dep")
        libpaths = self._linkfiles[1:]
        self._cmd.link_libs(ipath=llvmfile, libpaths=libpaths,
                            opath=pre_builtin_path)

        # Link library with the builtin modules
        linked_path = self._track_temp_file("linked-path")
        self.link_builtins(ipath=pre_builtin_path, opath=linked_path)

        # Optimize
        opt_path = self._track_temp_file("optimized-llvm-ir")
        self._cmd.optimize(ipath=linked_path, opath=opt_path)

        if config.DUMP_OPTIMIZED:
            with open(opt_path, 'rb') as fin:
                print(fin.read().decode('ascii'))

        with open(opt_path, 'rb') as fin, open("3_llvmir_optimized.ll","w") as outfile:
            outfile.write(fin.read().decode('ascii'))

        # Compile the llvm to HSAIL
        hsail_path = self._track_temp_file("create-hsail")
        self._cmd.generate_hsail(ipath=opt_path, opath=hsail_path)

        with open(hsail_path, 'rb') as fin, open("4_compiled.hsail","w") as outfile:
            outfile.write(fin.read().decode('ascii'))

        # Compile the llvm to BRIG
        brig_path = self._track_temp_file("create-brig")
        self._cmd.generate_brig(ipath=opt_path, opath=brig_path)

        with open(brig_path, 'rb') as fin, open("5_compiled.brig","wb") as outfile:
            outfile.write(fin.read())

        # link
        end_brig_path = self._track_temp_file("linked-brig")
        self._cmd.link_brig(ipath = brig_path, opath=end_brig_path)

        with open(end_brig_path, 'rb') as fin, open("6_compiled_linked.brig","wb") as outfile:
            outfile.write(fin.read())

        self._finalized = True

        # Read HSAIL
        with open(hsail_path, 'rb') as fin:
            hsail = fin.read().decode('ascii')

        # Read BRIG
        with open(end_brig_path, 'rb') as fin:
            brig = fin.read()

        if config.DUMP_ASSEMBLY:
            print(hsail)

        return namedtuple('FinalizerResult', ['hsail', 'brig'])(hsail, brig)