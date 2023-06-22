<!---
MIT License

Copyright (c) 2023 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->
# Feedback and Reporting Issues

We are looking forward to get your positive or negative feedback --- especially the negative feedback.
In particular, we are interested to learn:

* What libraries that you need are missing?
* What changes to the existing interfaces would make your life easier?
* What else blocks you or could be improved?

## Reporting Issues

We use [GitHub Issues](https://github.com/AMD-AI/hip-python/issues) to track public **bugs** and **enhancement requests**.

If you have found an issue, please check the [HIP Python documentation](https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html) 
to see if it hasn't already been resolved in the latest version of HIP Python.

### Bugs

Please follow the template below to report bugs that you found in HIP Python:

1. Description: ***Please be clear and descriptive***
2. How to Reproduce the issue:
* Hardware Information:
* OS version/docker environment:
* HIP Python package version and/or release branch that you have used:
* Expected behavior:
* Actual behavior:
3. Any additional information:

### Enhancement Requests

Please follow the template below to request any enhancement for HIP Python:

1. Description: ***Please be clear and descriptive***
2. Value and Motivation:
* Feature and functionalities enabled:
* Any alternatives:
3. Any additional information:

Authors must set labels (and assign a miliestone) according to their own understanding.

Other contributors can change these values if they disagree. That being said, 
adding a small comment explaining the motivation is highly recommended. 
In this way, we keep the process flexible while cultivating mutual understanding.

:::{note}

Most likely, labels like "bug", "feature" or "complexity*" 
will not cause discussions while others like "value*" or "urgency*" might
do so.
:::

## Creating Pull Requests

No changes are allowed to be directly committed to the `dev` and release
branches of the HIP Python repository. All authors are required to 
develop their change sets on a separate branch and then create 
a pull request (PR) to merge their changes into the respective branch.

Once a PR has been created, a developer must choose two reviewers 
to review the changes made. The first reviewer should be a 
technical expert in the portion of the library that the changes 
are being made in. The second reviewer should be a peer reviewer. This reviewer 
can be any other HIP Python developer.

## Responsibility of the Author

The author of a PR is responsible for:

 * Writing clear, well documented code
 * Meeting expectations of code quality
 * Verifying that the changes do not break current functionality
 * Writing tests to ensure code coverage
 * Report on the impact to performance

## Responsibility of the Reviewer

Each reviewer is responsible for verifying that the changes are 
clearly written in keeping with the coding styles of the library, 
are documented in a way that future developers will be able to 
understand the intent of the added functionality, and will 
maintain or improve the overall quality of the codebase.

Reviewer's task checklist:

* [ ] Has the PR passed necessary CI?
* [ ] Does the PR consist of a well-organized sequence of small commits, each of which is designed to make one specific feature or fix (and ideally should be able to pass CI testing)?
* [ ] Does the PR only include a reviewable amount of changes? Or it is a  consolidation of already reviewed small batches? e.g. break it into smaller testable and reviewable tasks instead of a huge chunk at once.
* [ ] Is PR sufficiently documented and it is easy to read and understand,  is it feasible for test and future maintainence? Do related docs already exist in the
[HIP Python documentation](https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html) if API or functionality has changed?
* [ ] For bugfixes and new features, new regression test created and included in CI, or some other holistic test pipeline?
* [ ] Is every PR associated with a ticket or issue number for tracking purposes?

## Passing CI

The most critical component of the PR process is the CI testing. 
All PRs must pass the CI in order to be considered for merger. 
Reviewers may choose to defer their review until the CI testing 
has passed. 

## The Review

During the review, reviewers will look over the changes and make 
suggestions or requests for changes.

In order to assist the reviewer in prioritizing their efforts, 
authors can take the following actions:

* Set the urgency and value labels.
* Set the milestone where the changes need to be delivered.
* Describe the testing procedure and post the measured effect of 
  the change.
* Remind reviewers via email if a PR needs attention.
* If a PR needs to be reviewed as soon as possible, explain to 
  the reviewers why a review may need to take priority.

### PRs affecting autogenerated code

In situations where your PR affects code that is autogenerated, the PR creation and review
should take place as described above, however the reviewers should additionally 
appoint a HIP Python developer to integrate the fix or enhancement into the code 
generator of HIP Python.

## Other Feedback/Requests

For other feedback or requests, please use this address:

```
hip-python.maintainer@amd.com
```