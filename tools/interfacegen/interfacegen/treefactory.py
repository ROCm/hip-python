# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

__author__ = "Advanced Micro Devices, Inc."

import sys

import logging

_log = logging.getLogger("interfacegen")

import clang.cindex

from .support.recipes import control

from . import cython


def from_libclang_translation_unit(
    translation_unit: clang.cindex.TranslationUnit,
    warn_mode=control.Warnings.WARN,
    backend=cython,
):
    """Create a tree from a libclang translation unit."""

    def first_child_cursor_of_kinds_(cursor: clang.cindex.Cursor, kinds: tuple):
        """Returns the first typeref child or None. Not recursive."""
        return next(
            (
                child_cursor
                for child_cursor in cursor.get_children()
                if child_cursor.kind in kinds
            ),
            None,
        )

    structure_types = {
        clang.cindex.CursorKind.STRUCT_DECL: backend.Struct,
        clang.cindex.CursorKind.UNION_DECL: backend.Union,
        clang.cindex.CursorKind.ENUM_DECL: backend.Enum,
    }
    anon_structure_types = {
        clang.cindex.CursorKind.STRUCT_DECL: backend.AnonymousStruct,
        clang.cindex.CursorKind.UNION_DECL: backend.AnonymousUnion,
        clang.cindex.CursorKind.ENUM_DECL: backend.AnonymousEnum,
    }

    def handle_top_level_cursor_(cursor: clang.cindex.Cursor, root: backend.Root):
        """Handle cursors whose parent is the cursor of kind TRANSLATION_UNIT."""
        nonlocal structure_types
        nonlocal warn_mode

        if cursor.kind in structure_types.keys():
            handle_top_level_record_or_enum_cursor_(cursor, root)
        elif cursor.kind == clang.cindex.CursorKind.TYPEDEF_DECL:
            handle_typedef_cursor_(cursor, root)
        elif cursor.kind == clang.cindex.CursorKind.VAR_DECL:
            if warn_mode in (control.Warnings.WARN, control.Warnings.ERROR):
                msg = (
                    f"VAR_DECL cursor '{cursor.spelling}' not handled (not implemented)"
                )
                if warn_mode == control.Warnings.WARN:
                    _log.warning(msg)
                else:
                    _log.error(f"ERROR: {msg}'")
                    sys.exit(2)
        elif cursor.kind == clang.cindex.CursorKind.MACRO_DEFINITION:
            root.append(backend.MacroDefinition(cursor, root))
        elif cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            typeref_cursor = first_child_cursor_of_kinds_(
                cursor, (clang.cindex.CursorKind.TYPE_REF,)
            )
            typeref = root.lookup_type_from_cursor(typeref_cursor)
            node = backend.Function(cursor, root, typeref=typeref)
            descend_into_child_cursors_(node)
            root.append(node)

    def handle_top_level_record_or_enum_cursor_(
        cursor: clang.cindex.Cursor, root: backend.Root
    ):
        """Handle a STRUCT_DECL/UNION_DECL cursor's STRUCT_DECL/UNION_DECL/ENUM_DECL child cursor.
        Other cursors are ignored.

        Note:
            In contrast to the `handle_nested_record_or_enum_cursor_`,
            this routine never creates `AnonymousStruct`, `AnonymousUnion`, `AnonymousEnum`
            instances, instead it sets a flag indicating that the node
            is from a typedef with anoymous inner node. This is mainly
            for debugging purposes. It is assumes that the name of the node
            gets overwritten when the respetive typedef is handled.
        """
        nonlocal structure_types

        if cursor.kind in structure_types:
            cls = structure_types[cursor.kind]
            node = cls(
                cursor, root, from_typedef_with_anon_child=(cursor.spelling == "")
            )
            descend_into_child_cursors_(node)
            root.append(node)

    def handle_typedef_cursor_(cursor: clang.cindex.Cursor, root: backend.Root):
        """Handle typedef cursors with respect to their children and type.

        Checks if the typedef has any STRUCT_DECL, UNION_DECL, ENUM_DECL, or TYPE_REF child cursor, which
        all indicate that there is already a node in the Root's child_nodes list for the inner type
        due to libclang's way of constructing the parse tree.

        In case of the former three, three different cases have to be handled:

        1. The inner type is anonymous.

           In this case, the previously inserted (anonymous) Struct/-Union/-Enum node has to
           be given a name that uses a `ctypedef struct <name>`/...
           instead of `cdef struct <name>`/...  when rendering Cython code,
           where `<name>` is the spelling of the `TYPEDEF_DECL` cursor. (FIXME Cython backend specific text)

        2. Inner type and typedef name are the same.

            In this case, no Typedef node is inserted as only `cdef struct <name>`/...  needs to be specified
            in the rendered Cython code.

        3. Inner type and typedef name differ.

            In this case a Typedef case is inserted that specifies a previously added
            Struct/Union/Enum as typeref argument.

        In case none of the listed four child cursors could be found,
        the routine checks if the cursor's type might be a typedefed function pointer.
        In this case, no Typedef node but a `TypedefedFunctionPointer` is inserted.
        """
        if backend.TypedefedFunctionPointer.match(cursor.type):
            _log.debug(
                f"handle_typedef_cursor_: typedefed function pointer: found {cursor.type.kind} with typedef name '{cursor.spelling}'"
            )
            node = backend.TypedefedFunctionPointer(cursor, root)
            descend_into_child_cursors_(node)  # post-order walk,
            root.append(node)
        elif backend.ConstantArray.match_typedefed_constantarray_of_basic_type(
            cursor.type
        ):
            _log.debug(
                f"handle_typedef_cursor_: typedefed constant array of basic type elements found: found {cursor.type.kind} with typedef name '{cursor.spelling}'"
            )
            node = backend.ConstantArray(
                cursor,
                root,
            )
            root.append(node)
        elif backend.Typedef.match_typedefed_basic_type(cursor.type):
            _log.debug(
                f"handle_typedef_cursor_: typedefed basic type: found {cursor.type.kind} with typedef name '{cursor.spelling}'"
            )
            node = backend.Typedef(cursor, root)
            root.append(node)
        elif backend.Typedef.match_typedefed_void(cursor.type):
            _log.debug(
                f"handle_typedef_cursor_: typedefed void: found {cursor.type.kind} with typedef name '{cursor.spelling}'"
            )
            node = backend.Typedef(cursor, root)
            root.append(node)
        elif backend.Typedef.match_typedefed_pointer(cursor.type):
            _log.debug(
                f"handle_typedef_cursor_: typedefed pointer type: found {cursor.type.kind} with typedef name '{cursor.spelling}'"
            )
            node = backend.Typedef(cursor, root)
            typeref_cursor = first_child_cursor_of_kinds_(  #
                cursor, (clang.cindex.CursorKind.TYPE_REF,)
            )  # TODO see if looking up the TYPE_REF cursor can be done via clang.cindex.
            if typeref_cursor is not None:
                node.typeref = root.lookup_type_from_cursor(typeref_cursor)
            root.append(node)
        elif backend.Typedef.match_typedefed_record_or_enum(
            cursor.type
        ):  # typedef of struct or union
            type_decl_cursor = cursor.underlying_typedef_type.get_declaration()  # FIX
            if not len(
                type_decl_cursor.spelling
            ):  # found anonymous struct/union/enum child
                _log.debug(
                    f"handle_typedef_cursor_: typedefed enum/record: found anonymous {type_decl_cursor.type.kind} cursor with typedef name '{cursor.spelling}'"
                )
                # in case of anon enum, replace the original node with the given one
                type_decl = root.lookup_type_from_cursor(type_decl_cursor)
                assert type_decl != None, backend.Node.render_cursor_location(cursor)
                assert isinstance(type_decl, (backend.Enum, backend.Record))
                assert type_decl._from_typedef_with_anon_child
                type_decl.overwrite_name(cursor.spelling)
                pass  # do not append typedef node
            elif (
                type_decl_cursor.spelling != cursor.spelling
            ):  # child with different name
                _log.debug(
                    f"handle_typedef_cursor_: typedefed enum/record: found {type_decl_cursor.type.kind} with name '{type_decl_cursor.spelling}' and typedef name '{cursor.spelling}'"
                )
                # update, append typedef node
                node = backend.Typedef(
                    cursor, root
                )  # quiet/silent creation depending on case
                node.typeref = root.lookup_type_from_cursor(type_decl_cursor)
                root.append(node)
            else:  # child with same name
                _log.debug(
                    f"handle_typedef_cursor_: typedefed enum/record: found {type_decl_cursor.type.kind} with name and typedef name '{type_decl_cursor.spelling}'"
                )
                pass  # do not append typedef node
        else:
            _log.warning(
                f"<{backend.tree.Node.render_cursor_location(cursor)}> Did not handle {cursor.type.kind} with typedef name '{cursor.spelling}'"
            )

    def handle_nested_record_or_enum_cursor_(
        cursor: clang.cindex.Cursor, parent: backend.Node
    ):
        """Handle a STRUCT_DECL/UNION_DECL cursor's STRUCT_DECL/UNION_DECL/ENUM_DECL child cursor.
        Other cursors are ignored.
        """
        nonlocal structure_types
        nonlocal anon_structure_types

        is_anonymous = cursor.spelling == ""

        if cursor.kind in structure_types:
            cls = structure_types[cursor.kind]
            cls_anon = anon_structure_types[cursor.kind]
            if is_anonymous:
                node = cls_anon(cursor, parent)
            else:
                node = cls(cursor, parent)
            descend_into_child_cursors_(node)
            parent.append(node)

    def handle_param_or_field_decl_cursor_(
        cursor: clang.cindex.Cursor, parent: backend.Node
    ):
        """Handle PARAM_DECL/FIELD_DECL cursors.

        First check if the cursor's type is anonymous function pointer.
        In this case emit an additional AnonymousFunctionPointer node.
        If there are further PARAM_DECL children of the given cursor, it visits
        them first before emitting an `AnonymousFunctionPointer` node.
        This guarantees that nested anoymous pointers are processed
        before constructing the `AnonymousFunctionPointer` node for the parent cursor.

        Note:
            AnonymousFunctionPointer instances must appear as child node of their
            parent in order to give them a unique index.
        """
        assert cursor.kind in (
            clang.cindex.CursorKind.PARM_DECL,
            clang.cindex.CursorKind.FIELD_DECL,
        )
        if backend.AnonymousFunctionPointer.match(cursor.type):
            typeref = backend.AnonymousFunctionPointer(cursor, parent)
            descend_into_child_cursors_(typeref)  # post-order walk
            parent.append(typeref)
        else:
            # FIXME prove robustness
            typeref_cursor = first_child_cursor_of_kinds_(
                cursor,
                (
                    clang.cindex.CursorKind.TYPE_REF,
                    clang.cindex.CursorKind.STRUCT_DECL,
                    clang.cindex.CursorKind.UNION_DECL,
                    clang.cindex.CursorKind.ENUM_DECL,
                ),
            )
            # TODO check that typeref is only None if no type is involved
            root = parent.get_root()
            typeref = root.lookup_type_from_cursor(typeref_cursor)
        if cursor.kind == clang.cindex.CursorKind.PARM_DECL:
            node = backend.Parm(cursor, parent, typeref=typeref)
        else:
            node = backend.Field(cursor, parent, typeref=typeref)
        parent.append(node)

    def descend_(cursor, parent=None):
        assert isinstance(parent, backend.Node)
        assert parent.cursor is not None
        parent_cursor = parent.cursor
        if parent_cursor.kind == clang.cindex.CursorKind.TRANSLATION_UNIT:
            handle_top_level_cursor_(cursor, parent)
        elif parent_cursor.kind in (
            clang.cindex.CursorKind.STRUCT_DECL,
            clang.cindex.CursorKind.UNION_DECL,
        ):
            handle_nested_record_or_enum_cursor_(cursor, parent)
        #
        if cursor.kind in (
            clang.cindex.CursorKind.PARM_DECL,
            clang.cindex.CursorKind.FIELD_DECL,
        ):
            handle_param_or_field_decl_cursor_(cursor, parent)

    def descend_into_child_cursors_(node: backend.Node):
        for child_cursor in node.cursor.get_children():
            descend_(child_cursor, node)

    root = backend.Root(translation_unit.cursor)
    descend_into_child_cursors_(root)
    return root
