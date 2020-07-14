import unittest

import jittor as jt

import neural_renderer as nr

class TestLighting(unittest.TestCase):
    
    def test_case1(self):
        """Test whether it is executable."""
        faces = jt.random([64, 16, 3, 3])
        textures = jt.random([64, 16, 8, 8, 8, 3])
        nr.lighting(faces, textures)

if __name__ == '__main__':
    unittest.main()



