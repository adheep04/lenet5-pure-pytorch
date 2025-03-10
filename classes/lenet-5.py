from torch import nn
import torch

from collections import OrderedDict

from convolution_layer import ConvolutionLayer
from tanh import TanhActivation
from avg_pooling import AvgPoolingLayer
from radial_basis_function import RadialBasisFunctionLayer


class LeNet_5(nn.Module):
    def __init__(self, config):  
        super().__init__()
        self.config = config
        self.A = 1.7159
        self.S = 2/3
        
        # define connection as a list of lists where the values of a list i correspond to 
        # the incoming channels of filter i's convolution in layer C3
        self.connections = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [0, 4, 5],
            [0, 1, 5],
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [0, 3, 4, 5],
            [0, 1, 4, 5],
            [0, 1, 2, 5],
            [0, 1, 3, 4],
            [1, 2, 4, 5],
            [0, 2, 3, 5],
            [0, 1, 2, 3, 4, 5],
        ]
        
        self.pipeline = nn.Sequential(OrderedDict([
            # (1, 32, 32) -> (6, 28, 28)
            ('C1', ConvolutionLayer(num_filters=6, filter_size=5, in_channels=1, efficient=config['efficient'])),
            ('tanh1', TanhActivation()),
                            
            # (6, 28, 28) -> (6, 14, 14)
            ('S2', AvgPoolingLayer(num_channels=6, efficient=True)),
            ('tanh2', TanhActivation()),
                    
            # (6, 14, 14) -> (16, 10, 10)
            ('C3', ConvolutionLayer(num_filters=16, filter_size=5, in_channels=6, efficient=config['efficient'], connections=self.connections)),
            ('tanh3', TanhActivation()),
                    
            # (16, 10, 10) -> (16, 5, 5)
            ('S4', AvgPoolingLayer(num_channels=16, efficient=config['efficient'])),
            ('tanh4', TanhActivation()),
                                                            
            # (16, 5, 5) -> (120, 1, 1)
            ('C5', ConvolutionLayer(num_filters=120, filter_size=5, in_channels=16, efficient=config['efficient'])),
            ('tanh5', TanhActivation()),
                        
            # (120, 1, 1) -> (84)
            ('F6', nn.Linear(in_features=120, out_features=84)),
            ('tanh6', TanhActivation()),
                                            
            # (84) -> (10,)
            ('RBF', RadialBasisFunctionLayer(weights=self.get_RBF_weights(), num_classes=10, size=84, efficient=config['efficient']))
        ]))
    
    def forward(self, x):
        x = self.pipeline(x)
        return x

    def get_RBF_weights(self):
        # stylized 12x7 bitmaps for digits 0-9
        digits = [
            [
                "       ",
                "  ***  ",
                " *   * ",
                "*     *",
                "*     *",
                "*     *",
                "*     *",
                "*     *",
                " *   * ",
                " ** ** ",
                "  ***  ",
                "       ",
            ],
            [
                "   **  ",
                "  ***  ",
                " * **  ",
                "   **  ",
                "   **  ",
                "   **  ",
                "   **  ",
                "   **  ",
                "   **  ",
                "   **  ",
                "  **** ",
                "       ",
            ],
            [
                " ***** ",
                " *   **",
                "     **",
                "     **",
                "    ** ",
                "   **  ",
                "  **   ",
                " **    ",
                " *     ",
                "**     ",
                "*******",
                "       ",
            ],
            [
                " ***** ",
                "**    *",
                "      *",
                "     * ",
                "    ** ",
                "  ***  ",
                "    ** ",
                "      *",
                "      *",
                "**   **",
                " ***** ",
                "       ",
            ],
            [
                "       ",
                "*     *",
                "**    *",
                "**    *",
                "*     *",
                "*******",
                "     **",
                "      *",
                "      *",
                "      *",
                "      *",
                "       ",
            ],
            [
                "       ",
                "*******",
                "*      ",
                "**     ",
                "**     ",
                "  **** ",
                "     **",
                "      *",
                "      *",
                "*    **",
                " ***** ",
                "       ",
            ],
            [
                " ***** ",
                "**     ",
                "*      ",
                "*      ",
                "****** ",
                "**   **",
                "*     *",
                "*     *",
                "*     *",
                "**    *",
                " ***** ",
                "       ",
            ],
            [
                "*******",
                "     **",
                "     **",
                "    ** ",
                "    *  ",
                "   **  ",
                "   *   ",
                "  **   ",
                "  **   ",
                "  *    ",
                "  *    ",
                "       ",
            ],
            [
                " ***** ",
                "**   **",
                "*     *",
                "**    *",
                " ***** ",
                "**   **",
                "*     *",
                "*     *",
                "*     *",
                "**   **",
                " ***** ",
                "       ",
            ],
            [
                " ***** ",
                "*     *",
                "*     *",
                "**    *",
                " ******",
                "      *",
                "      *",
                "      *",
                "      *",
                "     **",
                "  **** ",
                "       ",
            ],
        ]
        
        bitmap = torch.empty(10, 12, 7)
        for d, digit in enumerate(digits):
            for j, row in enumerate(digit):
                for i, char in enumerate(row):
                    if char == "*":
                        bitmap[d, j, i] = 1.0
                    else:
                        bitmap[d, j, i] = -1.0
                        
        # validate no NaN's
        assert not torch.isnan(bitmap).any(), f"digit bitmap isn't being populated correctly"
        assert bitmap.shape == (10, 12, 7), f"incorrect digit bitmap dimensions"
                        
        # flatten to (10, 84)
        return bitmap.flatten(start_dim=1, end_dim=2)

    
