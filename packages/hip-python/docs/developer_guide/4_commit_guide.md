# Commit Guidelines

HIP Python development is adopting
[Commitizen](https://commitizen-tools.github.io/commitizen/) to enforce
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) and
implement automatic release tagging based on commit messages.

## Commit message rules

A commit message should like like this:
```text
<type>[optional scope]: <subject>

[optional body]

[optional footer(s)]
```
Commit messages start with a header line that notes the type of change, followed
by an optional list of scopes, then the description of the change.

Type must be one of:

- **`build`**: Changes that affect the build system or external dependencies
  (example scopes: `gulp`, `broccoli`, `npm`)
- **`ci`**: Changes to our CI configuration files and scripts
- **`docs`**: Documentation only changes
- **`feat`**: A new feature, *corresponds to a minor version bump*
- **`fix`**: A bug fix, *correspond to a patch version bump*
- **`perf`**: A code change that improves performance
- **`refactor`**: A code change that neither fixes a bug nor adds a feature
- **`style`**: Changes that do not affect the meaning of the code (white-space,
  formatting, missing semi-colons, etc)
- **`test`**: Adding missing tests or correcting existing tests

The **scope** if included should be the area the change affects in parentheses
i.e. `(theme)` for theming changes or `(deps)` for changes to the list of
dependencies.
Multiple scopes can be given and `(all)` may be used if change affects the
entire project.

The **subject** should be a succinct description of the change in present imperative
tense (i.e. *update wording* not *updated wording*)

A longer description can be included in the **body**, including the motivation
of the change and comparison with the previous version. Comparisons with the
prior state if shouldn't detail literal changes, those are recorded by the git
history, but the semantic or higher level effect of the modifications.

The footer contains information about **Breaking Changes** and references to
GitHub issues that the change closes or relates to.
Breaking changes should be noted with the words `BREAKING CHANGE:` followed by
a space and the description of the change. Including a breaking change in a
commit will result in a major version bump of the next release including such
commit.

For a more detailed description of conventional commits please refer to the
specification linked above.

## Committing with Commitizen

Commitizen contains a utility to walk through making a commit following the
rules above.

### Installation

Commitizen is marked as a "development" dependency of `rocm-docs-core,`
installable via `pip install .[development]` from the root of the repository.

### Usage

Run `cz commit` or `cz c` to start making a commit after staging the files to
be added. Commitizen will ask for each part of the commit message in turn.

Refer to the documentation of Commitizen (linked above) on other options, and
other features of Commitizen.