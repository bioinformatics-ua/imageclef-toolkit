# ImageCLEF concept count tool

This is a simple tool for analyzing and processing text files containing lists of concepts.

## Building & Using

Rust is required for building this tool. [Install Rust here](https://www.rust-lang.org/en-US/install.html) with the latest stable toolchain. Then:

```sh
cargo run --release
```

To extract the full concept count from the 2018 edition of the data set, for example:

```sh
cargo run --release -- vocabulary ConceptDetectionTraining2018-Concepts.csv -o concepts.csv
```
