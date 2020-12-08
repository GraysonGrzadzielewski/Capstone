if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
    description="            Capstone Repository 2020\n"+
"Brian Crutchley:            bcrutc01@rams.shepherd.edu\n"+
"Grayson Grzadzielewski:     ggrzad01@rams.shepherd.edu\n\n"+
"To Run the NEAT Implementation, Include only the '--neat' Parameter\n\n"+
"To Run the Curiosity Implementation, Include '--curiosity' and '--name' Parameters AT LEAST\n",
    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--test', action='store_true', help='Run the test script for the chosen implementation', default=False)
    # NEAT arguments
    parser.add_argument('--neat', action='store_true', help='Pass arguments to NEAT implementation', default=False)
    # Curiosity arguments
    parser.add_argument('--curiosity', action='store_true', help='Pass arguments to Curiosity implementation', default=False)
    parser.add_argument('--name', type=str, help='Name of files', default='')
    parser.add_argument('--load_model', type=bool, help='Saved model to load', default=False)
    parser.add_argument('--exp', action='store_true',
                        help='Run with --batch number of best runs selected from the trajectory.\n'+
                        'WARNING: Don\'t change this setting if loading an existing model!',
                        default=False
                        )
    parser.add_argument('--steps', type=int, help='Time steps in trajectory', default=2450)
    parser.add_argument('--batch', type=int, help='Size of the batch, use with --exp', default=32)
    parser.add_argument('--trajectory', type=int, help='Size of trajectory', default=32)
    parser.add_argument('--minibatch', type=int, help='Size of minibatch', default=128)
    parser.add_argument('--epochs', type=int, help='Number if epochs', default=4)
    parser.add_argument('--device', type=str, help='Pytorch device to use during training', default='cpu')
    parser.add_argument('--env', type=str, help='Game environment to run', default='SuperMarioBros-Nes')
    args = parser.parse_args()

    if args.neat:
        if not args.test:
            import run_neat
            run_neat.main()
        else:
            import neat_test
            neat_test.main()
    elif args.curiosity:
        if not args.test:
            import run_curiosity
            run_curiosity.main(args)
        else:
            import curiosity_test
            curiosity_test.main(args.name)
    else:
        raise Exception('Either neat or curiosity needs to be specified as the implementation')
