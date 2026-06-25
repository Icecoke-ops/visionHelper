#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visionHelper 统一命令行入口。

使用方式：python scripts/vh.py <subcommand> <action> [options]
"""

import sys

from scripts.cli import main

if __name__ == "__main__":
    sys.exit(main())
