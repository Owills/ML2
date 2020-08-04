class basicString:
    def __init__(self, string="..."):
        self.str = string

    def inputString(self):
        self.str = input("Enter String: ")

    def printUpper(self):
        print(self.str.upper())


x = basicString("")
x.inputString()
x.printUpper()
