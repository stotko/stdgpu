# Contributing

Thank you for taking the time to contribute to the project. The following set of guidelines will help you with your contribution and explain the review process.


## Reporting Bugs / Proposing New Features

If you discover a bug or want to propose a new feature or enhancement to the code or documentation, please check whether the problem has already been reported in the [Issues](https://github.com/stotko/stdgpu/issues).

If not, open a new issue and state the problem. Please provide a clear summary of the problem, what behavior you have expected and what behavior you have actually observed. If possible, create a *Minimal Reproducable Example* that demonstrates the problem.


## Submitting Changes

We also highly welcome code contributions via pull requests. Note that your changes - after acceptance - will be offered under the **Apache 2.0** license. For more information, see the [LICENSE](https://github.com/stotko/stdgpu/blob/master/LICENSE).

To create and submit your changes, follow the standard *Fork & Pull Request Workflow*:

1. Fork the project and switch to a new suitably named branch.
2. Create one or more commits that reflect the changes you have made. Each commit should be self-contained, atomic and buildable. Therefore, split multiple features into different commits and include fixups in the related commit instead of creating a new one. If you add new functionality, please also add related tests.
3. Test the changes on your local machine. For this, the provided scripts will help you to build the code and run the unit tests. See the [README](https://github.com/stotko/stdgpu/blob/master/README.md) for details.
4. Push the branch to your fork.
5. Open a new pull request and summarize the problem and your solution. If there is a related issue, please mention it too.

Once you have submitted the pull request, your changes will be reviewed. You will receive feedback in two different forms:

1. **Automatic review**. Each pull request will be automatically tested using *Continuous Integration* tools. If a test fails, take a look at the error and fix the problem.
2. **Manual human review**. In addition, your code will be manually reviewed by the project maintainers. Incorporate their feedback to improve the quality of your contribution.

After the review is complete and all tests pass, your pull request will be accepted and merged.
