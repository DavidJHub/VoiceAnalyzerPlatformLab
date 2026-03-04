class logger:
    def __init__(self, *files):
        self.files = files

    def write(self, message):
        for file in self.files:
            safe_message = message.encode('ascii', errors='replace').decode('ascii')
            file.write(safe_message)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()


