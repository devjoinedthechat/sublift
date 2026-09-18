# Security

sublift is a statistical library. It does not open sockets, run a server, or execute anything
it reads. The realistic risks are narrower than for most packages, and worth stating plainly.

## What counts as a vulnerability

- Code execution reachable from data passed to sublift (a crafted DataFrame, a pickled object).
- A supply-chain problem in how sublift is built or published.
- **A silent correctness failure in an estimator.** If sublift can be made to report a
  confident interval around a wrong number without warning, treat it as a security issue and
  report it the same way. Someone will ship a pricing decision on that number.

## Reporting

Open a [private security advisory](https://github.com/devjoinedthechat/sublift/security/advisories/new).
Please do not open a public issue first. You can expect an acknowledgement within a week.

For ordinary bugs — including estimators that are wrong in ways that are *visible* — a normal
public issue is the right place, and is very welcome.

## Supported versions

Pre-1.0: only the latest release is supported. Fixes land on `main` and in the next release.
