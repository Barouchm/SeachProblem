class Agent(object):
    """description of class"""
    def __init__(self,x_pos,y_pos, sensors):#at the moment does not work
        self.x_pos=x_pos
        self.y_pos=y_pos
        self.sensors = sensors
    def __init__(self,x_pos,y_pos):
        self.x_pos=x_pos
        self.y_pos=y_pos
    def setSensots(sensors):
        self.sensors = sensors
    def __repr__(self):
        return "agent "+str(self.x_pos)+" "+str(self.y_pos)
    def pos(self):
        return (x_pos,y_pos)




