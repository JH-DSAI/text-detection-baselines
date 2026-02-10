#!/usr/bin/env python

import click


@click.command(context_settings={"show_default": True})
def main():
    print("Hello, world!")


if __name__ == "__main__":
    main()
