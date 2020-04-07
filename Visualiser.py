from pscript import window
from flexx import flx
class NetworkVisualiser(flx.CanvasWidget):
    _current_node = flx.Property(None, settable=True)
    inputs = flx.ListProp(settable=True)
    neurons = flx.ListProp(settable=True)
    
    def init(self):
        self.ctx = self.node.getContext('2d')
        self.xx = [0.3, 0.3, 0.70, 0.60, 0.50, 0.40, 0.10]#, 0.23, 0.61, 0.88]
        self.yy = [0.33, 0.66, 0.90, 0.60, 0.90, 0.70, 0.55]#, 0.19, 0.11, 0.38]
        self.names = []
        self.node_width = 25
        i = [True, False]
        self._mutate_inputs(i)
        self._mutate_neurons(i)
        self.buttonlabel= flx.Label(text='dgfhdfgh')

    @flx.reaction('pointer_down')
    def _on_pointer_down(self, *events):
        for ev in events:
            w, h = self.size
            # Get closest point
            closest, dist = -1, 999999
            for i in range(len(self.xx)):
                x, y = self.xx[i] * w, self.yy[i] * h
                d = ((x - ev.pos[0]) ** 2 + (y - ev.pos[1]) ** 2) ** 0.5
                if d < dist:
                    closest, dist = i, d
            # Did we touch it or not
            if dist < self.node_width:
                i = closest
                self._set_current_node(i)
           

    @flx.reaction('pointer_up')
    def _on_pointer_up(self, *events):
        self._set_current_node(None)

    @flx.reaction('pointer_move')
    def _on_pointer_move(self, *events):
        ev = events[-1]
        if self._current_node is not None:
            i = self._current_node
            w, h = self.size
            self.xx[i] = ev.pos[0] / w
            self.yy[i] = ev.pos[1] / h
            self.update()
            
    @flx.emitter
    def key_down(self, e):
        return self._create_key_event(e)
    
    @flx.reaction('size', '_current_node', 'inputs', 'neurons')
    def update(self, *events):
        self.draw()
        
    @flx.action
    def update_inputs(self, i):
         self.set_inputs(i)       
          
    @flx.action
    def update_neurons(self, i):
         self.set_neurons(i)    
    
    @flx.action
    def draw(self, inputs, Net):
        ctx = self.ctx
        w, h = self.size
        ctx.fillStyle = "white"#"0d053b"
        #ctx.strokeStyle = "black"
        ctx.fillRect(0, 0, w, h)
        ctx.fillStyle = "red"
        ctx.strokeStyle = "red"
        #Draw rectangles for input signals
        
        for i in range(len(self.inputs)):
            
            if(self.inputs[i]):
                ctx.fillStyle = "yellow"
            else:
                ctx.fillStyle = "blue"
            
            
            posx = w * 0.1
            posy = (h/(len(self.inputs) + 1)) * (i + 1)
            ctx.fillRect(posx, posy, 25, 45)    
            #print(self.inputs[i])
        
        # Get coordinates
        xx = [x * w for x in self.xx]
        yy = [y * h for y in self.yy]
        
        # Draw nodes
        ctx.strokeStyle = '#000'
        ctx.lineWidth = 2
        for i in range(0, len(self.neurons)):
            if(self.neurons[i][0]):
                ctx.fillStyle = 'yellow'
            else:
                ctx.fillStyle = '#acf'
                
            ctx.beginPath()
            ctx.arc(xx[i], yy[i], self.node_width, 0, 6.2831)
            ctx.fill()
            ctx.stroke()
            
            ctx.strokeStyle = "black"
            ctx.font = '25px Calibri';
            ctx.strokeText(self.neurons[i][1], xx[i], yy[i]);


        # Draw lines
        #for i in range(1, len(xx)-2):
        #    ctx.lineCap = "round"
        #    ctx.lineWidth = 3
        #    ctx.strokeStyle = '#008'
         #   
         #   ctx.beginPath()
        #    lineto = ctx.lineTo.bind(ctx)
        #    
         #   lineto(xx[i+0], yy[i+0])
         #   lineto(xx[i+1], yy[i+1])
         #   ctx.stroke()