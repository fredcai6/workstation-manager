#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from workstation_manager.cli import main
sys.exit(main())
