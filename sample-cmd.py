import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=2)
    parser.add_argument("--model_file",default="model.pt")
    args = parser.parse_args()
    print(args)
    print(args.epochs)
    print(args.model_file)

if __name__ == "__main__":
    main()
