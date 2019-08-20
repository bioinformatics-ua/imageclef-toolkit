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

This tool supports all ground truth files from the 2017, 2018 and 2019 editions of the Caption challenge.

Use the `--help` flag to see the full list of subcommands or to know how to use a specific command:

```sh
cargo run -- vocabulary --help
```
