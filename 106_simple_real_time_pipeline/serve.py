
def mult_by_two(event):
    event["number"] *= 2
    return event

class MultByX:
    def __init__(self, x):
        self.x = x
        
    def do(self, event):
        event["number"] *= self.x
        return event
