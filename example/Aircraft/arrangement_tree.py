from marilib.aircraft import Arrangement
import numpy as np
import matplotlib.pyplot as plt
from anytree import Node, RenderTree, AsciiStyle, LevelOrderGroupIter

ARRANGEMENT_DICT={
          "body_type" :           ["fuselage", "blended"   , ""         , ""             , ""       , ""    , ""     ],
          "wing_type" :           ["classic" , "blended"   , ""         , ""             , ""       , ""    , ""     ],
          "wing_attachment":      ["low"     , "high"      , ""         , ""             , ""       , ""    , ""     ],
          "stab_architecture":    ["classic" , "t_tail"    , "h_tail"   , ""             , ""       , ""    , ""     ],
          "tank_architecture":    ["wing_box", "piggy_back", "pods"     , ""             , ""       , ""    , ""     ],
          "number_of_engine":     ["twin"    , "quadri"    , "hexa"     , ""             , ""       , ""    , ""     ],
          "nacelle_attachment" :  ["wing"    , "rear"      , "pods"     , ""             , ""       , ""    , ""     ],
          "power_architecture":   ["tf"      , "tp"        , "ef"       , "ep"           , "pte"    , "extf", "exef" ],
          "power_source" :        ["fuel"    , "fuel_cell" , "battery"  , ""             , ""       , ""    , ""     ],
          "fuel_type":            ["kerosene", "methane"   , "liquid_h2", "Compressed_h2", "battery", ""    , ""     ]
          }

class ArrangementTree(Node):
    """A custom anytree.Node object to describe all feasible arrangement choices"""
    def __init__(self,**kwargs):
        """ Keep only the branches of the tree passing through the selected node(s)
        :param **kwargs: the Arrangement settings that are set to a desired value. Example::
                tree = ArrangementTree(wing_type='classic')
        :return: the root node of the tree (anytree.Node)
        """
        super().__init__("Arrangement")

        # build a dict of 'fixed choices'
        fixed_nodes = dict.fromkeys(ARRANGEMENT_DICT.keys())    # default init
        fixed_nodes['body_type'] = 'fuselage'  # TODO : in Marilib 2.0, the body_type is always fixed to 'fuselage'
        fixed_nodes['wing_type'] = 'classic'  # TODO : in Marilib 2.0, the wing_type is always fixed to 'classic'
        for key,val in kwargs.items():
            if key not in ARRANGEMENT_DICT.keys():
                msg = "Arrangement has no entry named %s" % key
                raise KeyError(msg)
            else:
                if val not in ARRANGEMENT_DICT[key]:
                    raise ValueError("Invalid value %s for entry %s" %(val,key))
                else:
                    fixed_nodes[key] = val

        # Construct the tree, step by step (depth by depth)
        for key,settings in ARRANGEMENT_DICT.items():  # iterate over all arrangement options
            if fixed_nodes[key] is not None:
                for leaf in self.leaves:  # iterate over the "leaves" of the tree
                    leaf.children = (Node(s) for s in settings if (len(s) > 0 and s is fixed_nodes[key]))
            else:
                for leaf in self.leaves:
                    leaf.children = (Node(s) for s in settings if len(s) > 0)

        self.filter_feasible()

    def filter_feasible(self):
        """
        TODO : filter for feasible arrangement choices
        :return:
        """


    def write_txt(self,filename,root_node):
        """Save the tree to text file
        :param root_node:
        :return:
        """
        with open(filname,"w") as f:
            content = ""
            for pre,_, node in RenderTree(root_node,style = AsciiStyle):
                content += "%s%s\n" %(pre, node.name)
            f.write(content)

    def path_of_leaf(self,index):
        leaf = self.leaves[index]
        return [str(node.name) for node in leaf.path[1:]]

#-------------------------------------------------
# Plot the table of Arrangement settings
#-------------------------------------------------

colLabels = [k for k in ARRANGEMENT_DICT.keys()]
cellText = [[p for p in options] for options in ARRANGEMENT_DICT.values()]
cellText = list(zip(*cellText))  # trick to transpose the 2D list

fig = plt.figure("Arrangement choice helper",figsize=(15,5))
ax = plt.subplot(111)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.axis("off")
tit = ax.set_title("Select your settings")
tab = ax.table(colLabels=colLabels, colColours=['g']*len(colLabels),
                cellText = cellText,rowLoc='center', cellLoc='center',
                bbox=[0,0,1,1])
tab.auto_set_font_size(False)
tab.set_fontsize = 12
for k, cell in tab._cells.items():
    cell.set_edgecolor("silver")

# create an axes for lines plot
ax1 = ax.twiny()
ax1.set_xlim([0,1])
ax1.set_ylim([0,1])
ax1.axis("off")

arrangement_dict = {}

def onclick(event):
    if tab.contains(event)[0]:
        for (row,col),cell in tab.get_celld().items(): # iterate over all cells
            if cell.contains(event)[0] and cell.get_text().get_text() is not "" and row>0: # find the new selected cell
                if cell.get_facecolor()==(1,1,1,1): # if white :
                    i=1
                    while True:  # reset white color for all cells in the column
                        try:
                            tab[i,col].set_facecolor((1,1,1,1))
                            i+=1
                        except KeyError:
                            break
                    cell.set_facecolor((1,0,0,0.5))  # set face color to red
                    arrangement_dict[tab[0, col].get_text().get_text()] = cell.get_text().get_text()  # add this setting
                else: # if not white, then reset to white and delete dict entry
                    cell.set_facecolor('w')
                    try:
                        del arrangement_dict[tab[0, col].get_text().get_text()]
                    except KeyError:
                        print("WARNING: KeyError '%s'" % tab[0, col].get_text().get_text())

                ax1.clear()
                update_tree()
                plt.draw()
                break


def update_tree():
    tree = ArrangementTree(**arrangement_dict)
    N_conf = len(tree.leaves)
    if N_conf <500:  # check for reasonable number of possible configurations
        tit.set_text("Number of configurations : %d" % N_conf)
        tit.set_color('k')
        for k,leaf in enumerate(tree.leaves):
            x,y = path_to_line(tree.path_of_leaf(k),len(cellText[0]),len(cellText))
            ax1.plot(x,y,'-k',lw=3,alpha=1./len(tree.leaves))
    else:
        tit.set_text("TOO MUCH CONFIGURATIONS TO PLOT : %d" % N_conf)
        tit.set_color("r")

def path_to_line(path,n_x,n_y):
    points = []
    for j,setting in enumerate(path):  # iterate over all nodes of this tree branch
        for i in range(1,n_y+1):  # iterate over all lines of the table
            if tab[i,j].get_text().get_text()==setting:
                points.append((j,i))
                break
    x,y = zip(*points)
    x = (0.5 + np.array(x)) / n_x
    y = 1 - (np.array(y) + 0.5) / (n_y + 1)
    return x,y

fig.canvas.mpl_connect('button_press_event', onclick)
plt.tight_layout()
plt.show()

