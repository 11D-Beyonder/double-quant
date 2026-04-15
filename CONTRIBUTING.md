# Contributing

This document currently focuses on the project's recommended release workflow.

## Release Philosophy

This project follows a release flow that is common in many open source Python libraries:

1. Feature work is merged through normal pull requests.
2. A maintainer prepares a dedicated release pull request.
3. The release pull request updates the package version.
4. After the release pull request is merged into `main`, the maintainer creates and pushes a Git tag.
5. GitHub Actions builds the package, creates a GitHub Release, and publishes the package to PyPI.

This keeps feature development separate from release operations and makes the release intent explicit.

## Versioning Policy

- Do not update `[project].version` in `pyproject.toml` inside normal feature or bugfix pull requests.
- Update the version only in a dedicated release pull request prepared by a maintainer.
- The Git tag must match the package version with a `v` prefix.

Examples:

- `pyproject.toml`: `0.0.1a3`
- Git tag: `v0.0.1a3`

## Release Pull Request

When preparing a new release, open a dedicated pull request that only contains release-related changes.

Typical changes:

- Update `version` in [`pyproject.toml`](pyproject.toml)
- Optionally update release notes, README examples, or other user-facing documentation

Recommended rules:

- Keep the release pull request focused and small
- Do not mix feature work with the release pull request
- Review the version bump like any other pull request

## Maintainer Release Steps

After the release pull request is merged into `main`, the maintainer performs the release:

1. Pull the latest `main`
2. Create a tag that matches the version in `pyproject.toml`
3. Push the tag to GitHub
4. Wait for the release workflow to finish
5. Confirm that the GitHub Release and PyPI package were published successfully

Example:

```bash
git switch main
git pull origin main
git tag v0.0.1a3
git push origin v0.0.1a3
```

## GitHub Actions Responsibilities

The release workflow should be triggered by pushing a version tag such as `v0.0.1a3`.

The workflow is expected to:

1. Check out the tagged revision
2. Verify that the Git tag matches the version in `pyproject.toml`
3. Build both the source distribution and wheel
4. Publish the package to PyPI
5. Create a GitHub Release for the same version

If the tag and `pyproject.toml` version do not match, the workflow should fail.

The repository's release workflow is defined in [`.github/workflows/release.yml`](.github/workflows/release.yml).

It currently uses:

- `uv build` to build the source distribution and wheel
- `uv publish` to publish to PyPI
- GitHub Actions `environment: pypi` to require approval before the publish job runs

## GitHub Environment Setup

Before the release workflow can safely publish to PyPI, configure a GitHub Actions environment named `pypi`.

Recommended setup:

1. Open the repository on GitHub
2. Go to `Settings`
3. Open `Environments`
4. Create a new environment named `pypi`
5. Add at least one required reviewer
6. Optionally enable `Prevent self-review`

This project expects the publish job to pause for approval before uploading to PyPI.

## PyPI Trusted Publisher Setup

This project uses PyPI Trusted Publishing instead of storing a long-lived `PYPI_TOKEN` in GitHub secrets.

In the PyPI project settings for `double-quant`, add a Trusted Publisher with the following values:

- Owner: `11D-Beyonder`
- Repository name: `double-quant`
- Workflow filename: `.github/workflows/release.yml`
- Environment name: `pypi`

Important:

- The workflow filename must exactly match the workflow file in this repository
- The environment name must exactly match the GitHub Actions environment
- No `PYPI_TOKEN` secret is required when Trusted Publishing is configured correctly

## Maintainer Release Checklist

Use this checklist for every release:

1. Merge all intended feature and fix pull requests into `main`
2. Open a dedicated release pull request
3. Update [`pyproject.toml`](pyproject.toml) `version`
4. Optionally update release notes or user-facing documentation
5. Merge the release pull request into `main`
6. Create and push a matching Git tag
7. Wait for the release workflow to start
8. Approve the `pypi` environment when GitHub asks for review
9. Confirm the package appears on PyPI
10. Confirm the GitHub Release was created with build artifacts attached

## Example Release Session

If `pyproject.toml` is updated to `0.0.1a3`, the maintainer release session should look like this:

```bash
git switch main
git pull origin main
git tag v0.0.1a3
git push origin v0.0.1a3
```

Expected result:

- GitHub Actions validates that the tag matches `pyproject.toml`
- GitHub Actions builds the package with `uv build`
- The publish job waits for approval in the `pypi` environment
- After approval, GitHub Actions publishes with `uv publish`
- GitHub creates a release for `v0.0.1a3`

## Summary

Use this rule of thumb:

- Feature PRs do not change `version`
- Release PRs do change `version`
- Tags trigger publishing

This keeps the workflow predictable, reviewable, and aligned with common open source release practices.
