# ImageCLEF concept detection submission merger

This program merges two or more submission files by concatenating the concepts.

**There are two conditions for a proper merge at the moment:**

1. The file ids must occur at the exact same order.
2. The concepts of each file are assumed to be exclusive.

These constraints exists to make the program as simple as it can be to fulfill current use cases.

## Using

Rust is required for building this tool. [Install Rust here](https://www.rust-lang.org/en-US/install.html) with the latest stable toolchain. Then:

```sh
cargo run --release
```

For example:

```sh
cargo run --release -- submission-1.csv submission-2.csv > submission-1-plus-2.csv
```
