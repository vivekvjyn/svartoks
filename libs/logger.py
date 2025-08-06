import os

class Logger:
    def __init__(self, run_id, width=100):
        self.log_dir = "logs"
        self.name = run_id
        self.width = width
        self.bar_active = False
        os.makedirs(self.log_dir, exist_ok=True)
        if os.path.exists(os.path.join(self.log_dir, self.name + ".txt")):
            os.remove(os.path.join(self.log_dir, self.name + ".txt"))

    def _remove_line(self):
        with open(os.path.join(self.log_dir, self.name + ".txt"), 'rb+') as logfile:
            logfile.seek(0, 2)
            end = logfile.tell()
            pos = end - 1
            while pos > 0:
                logfile.seek(pos)
                if logfile.read(1) == b'\n':
                    break
                pos -= 1
            logfile.truncate(pos+1 if pos > 0 else 0)

    def message(self, message):
        self.bar_active = False
        with open(os.path.join(self.log_dir, self.name + ".txt"), "a") as logfile:
            logfile.write(message + "\n")
            print(message)

    def progress_bar(self, progress, total):
        with open(os.path.join(self.log_dir, self.name + ".txt"), "a") as logfile:
            if not self.bar_active:
                self.bar_active = True
            else:
                self._remove_line()
                print("\r", end="")
            logfile.write(f"|{'=' * int(self.width * progress / total)}{' ' * (self.width - int(self.width * progress / total))}|")
            print(f"|{'=' * int(self.width * progress / total)}{' ' * (self.width - int(self.width * progress / total))}|", end="")
            if progress == total:
                self.bar_active = False
                logfile.write("\n")
                print("\n", end="")

    def heading(self, heading):
        with open(os.path.join(self.log_dir, self.name + ".txt"), "a") as logfile:
            spaces = (self.width - len(heading)) // 2 * " "
            logfile.write(f'\n{spaces}{heading}\n\n')
            print(f'\n{spaces}{heading}\n')
