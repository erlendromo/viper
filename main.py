"""Single entry point for every ViPeR command.

    python main.py --help
    python main.py anyloc --help
    python main.py anyloc --db-imgs ... --queries ... --json ... --out runs/anyloc
"""
import argparse

from viper.anyloc import command as anyloc
from viper.megaloc import command as megaloc
from viper.salad import command as salad
from viper.siglip import command as siglip
from viper.training import command as train
from viper.utils import compare, pose_map, summary

# command name -> module with HELP, add_arguments(parser) and run(args)
COMMANDS = {
    "anyloc": anyloc,
    "salad": salad,
    "megaloc": megaloc,
    "train": train,
    "siglip": siglip,
    "compare": compare,
    "summary": summary,
    "pose-map": pose_map,
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="main.py", description="ViPeR: visual place recognition of IHM photographs in a ship scan.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    for name, module in COMMANDS.items():
        sub = commands.add_parser(name, help=module.HELP, description=module.__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
        module.add_arguments(sub)
        sub.set_defaults(run=module.run)
    
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
