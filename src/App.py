import Alice
from Pages import homepage

def main():

    app = Alice.App()
    home = Alice.Page(app, homepage.render)
    app.register("homepage", home)
    app.render()

if __name__ == "__main__":
    main()