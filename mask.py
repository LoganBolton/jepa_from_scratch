import torch 
import random

class MaskData():
    def __init__(self, num_blocks, grid_size):
        self.grid_size = grid_size
        self.num_blocks = num_blocks
        self.TARGET_ASPECT_MIN = 0.75
        self.TARGET_ASPECT_MAX = 1.5
        self.TARGET_SCALE_MIN = 0.15
        self.TARGET_SCALE_MAX = 0.2
        
        self.CONTEXT_ASPECT_MIN = 0.85
        self.CONTEXT_ASPECT_MAX = 1.0
        self.CONTEXT_SCALE_MIN = 0.85
        self.CONTEXT_SCALE_MAX = 1.0
        
        
    def get_block(self, grid_size, scale_min, scale_max, aspect_min, aspect_max):
        indices = []
        max_keep = random.uniform(scale_min, scale_max) * grid_size * grid_size
        aspect = random.uniform(aspect_min, aspect_max)
        width = int(min(round((max_keep / aspect) ** 0.5), grid_size))
        height = int(min(round((max_keep * aspect) ** 0.5), grid_size))
        
        curr_x = random.randint(0, grid_size-width)
        curr_y = random.randint(0, grid_size-height)
        
        for y in range(height):
            for x in range(width):
                flat = (curr_y+y) * grid_size + (curr_x+x)        
                indices.append(flat)
        return indices
                
    def get_indices(self):
        context_idxs = self.get_block(self.grid_size, 
                                     self.CONTEXT_SCALE_MIN, self.CONTEXT_SCALE_MAX, self.CONTEXT_ASPECT_MIN, self.CONTEXT_ASPECT_MAX)
        context_set = set(context_idxs)
        
        target_blocks = []
        for _ in range(self.num_blocks):
            target_idxs = self.get_block(self.grid_size, 
                                    self.TARGET_SCALE_MIN, self.TARGET_SCALE_MAX, self.TARGET_ASPECT_MIN, self.TARGET_ASPECT_MAX)
            for idx in target_idxs:
                if idx in context_set:
                    context_set.remove(idx)
            target_blocks.append(target_idxs)
            
        return list(context_set), target_blocks
            