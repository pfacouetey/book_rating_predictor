# CHANGELOG


## v0.2.0 (2025-04-15)

### Features

- Add linear models
  ([`fe367fc`](https://github.com/pfacouetey/book_rating_predictor/commit/fe367fc3d4ed96c349be3d30d488ccea8c9049be))

- **models**: Add linear models
  ([`ee4b306`](https://github.com/pfacouetey/book_rating_predictor/commit/ee4b3069eb26bfcefe63eeea44fd17bb3ff4c74d))

### Refactoring

- Group imports from same package
  ([`71135a8`](https://github.com/pfacouetey/book_rating_predictor/commit/71135a8c4a328a15905ea5ab4b39270172ff7e78))

Imports for the same package are scattered and not grouped together. It is recommended to keep the
  imports from the same package together. It makes the code easier to read.

- Put docstring into a single line
  ([`55f93d3`](https://github.com/pfacouetey/book_rating_predictor/commit/55f93d39cae9ae0eeb0f7a10a587cbeccf3e7d8d))

If a docstring fits in a single line (72 characters according to PEP8), it is recommended to have
  the quotes on the same line.

- Remove blank lines after docstring
  ([`a97ae86`](https://github.com/pfacouetey/book_rating_predictor/commit/a97ae86f7f08243856bc4c911299069680b59ac2))

There shouldn't be any blank lines after the function docstring. Remove the blank lines to fix this
  issue.

- Remove unnecessary `del` statement from local scope
  ([`84ebd3a`](https://github.com/pfacouetey/book_rating_predictor/commit/84ebd3acccce679cdbcd420ef257c0defd023a11))

Passing a local variable to a `del` statement results in that variable being removed from the local
  namespace. When exiting a function all local variables are deleted, so it is unnecessary to
  explicitly delete variables in such cases.

- Remove unnecessary comprehension
  ([`2da1ef9`](https://github.com/pfacouetey/book_rating_predictor/commit/2da1ef9b9a4a6d20750f2cac61def3092b118e9b))

The built-in function being used does not require comprehension and can work directly with a
  generator expression.


## v0.1.0 (2025-02-02)

### Features

- **loading-data-engineering-analysis**: Load data using the right format, proceed to some data
  engineering, then analyze results
  ([`2f9eb60`](https://github.com/pfacouetey/book_rating_predictor/commit/2f9eb6002d092449d4e7fe6a85afbbc66ae6c9eb))


## v0.0.0 (2025-01-04)
