class ProgressBar():
    def __init__(self, max_value, fractions=20):
        self.max_value = max_value
        self.current_value = 0
        self.fractions = fractions

    def draw(self):
        progress = (self.current_value * self.fractions) // self.max_value
        remaining = self.fractions - progress
        return f'[{progress * "#"}{remaining * "-"}] {self.current_value}/{self.max_value}'

    def next(self):
        if (self.current_value < self.max_value):
            self.current_value += 1
        return self.draw()

    def display(self):
        print(self.next(), end='\r')
        if (self.current_value == self.max_value):
            print('\nDone!')
