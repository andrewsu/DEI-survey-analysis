class Logger:
    def __init__(self):
        self.set_file("out/log.txt")
        self.data = ""
    
    def set_file(self, file: str):
        self.file = file
        with open(self.file, "w") as f:
            f.write("")

    def log_data(self, data: str):
        self.data += data + "\n"

    def write_log(self):
        with open(self.file, "a") as f:
            f.write(self.data)
        self.data = ""
