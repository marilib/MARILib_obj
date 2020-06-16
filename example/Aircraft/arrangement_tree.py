from marilib.aircraft import Arrangement
import numpy as np
import matplotlib.pyplot as plt
from anytree import Node, RenderTree, AsciiStyle, LevelOrderGroupIter

ARRANGEMENT_DICT={
# TODO : in Marilib 2.0, the body_type is always fixed to 'fuselage'
          "body_type" :           ["fuselage", ""          , ""         , ""             , ""       , ""    , ""     ],
# TODO : in Marilib 2.0, the wing_type is always fixed to 'classic'
          "wing_type" :           ["classic" , ""          , ""         , ""             , ""       , ""    , ""     ],
          "wing_attachment":      ["low"     , "high"      , ""         , ""             , ""       , ""    , ""     ],
          "stab_architecture":    ["classic" , "t_tail"    , "h_tail"   , ""             , ""       , ""    , ""     ],
          "tank_architecture":    ["wing_box", "piggy_back", "pods"     , ""             , ""       , ""    , ""     ],
          "number_of_engine":     ["twin"    , "quadri"    , "hexa"     , ""             , ""       , ""    , ""     ],
          "nacelle_attachment" :  ["wing"    , "rear"      , "pods"     , ""             , ""       , ""    , ""     ],
          "power_architecture":   ["tf"      , "tp"        , "ef"       , "ep"           , "pte"    , "extf", "exef" ],
          "power_source" :        ["fuel"    , "fuel_cell" , "battery"  , ""             , ""       , ""    , ""     ],
          "fuel_type":            ["kerosene", "methane"   , "liquid_h2", "compressed_h2", "battery", ""    , ""     ]
          }

"""The INCOMPATIBILITY_DICT stores all 'previous' incompatibilities for a given setting.
The term "previous" refers to the order of the options as daclared in the ARRANGEMENT_DICT.
For example, the option `nacelle_attachment` can not be `"rear", when the `stab_architecture` is set to `"classic"`
or `"h_tail"`, because MARILIB does not handle such cases.

>>> INCCOMPATIBILITY_DICT['nacelle_attachment']['rear'] = {'stab_architecture': ["classic","h_tail"]}
"""
INCOMPATIBILITY_DICT = {
    "body_type": None,
    "wing_type": {
        "blended": {
            "body_type": ["fuselage"]
        },
        "classic": {
            "body_type": ["blended"]
        }
    },
    "wing_attachement": None,
    "stab_architecture": None,
    "tank_architecture": {
        "piggy_back": {
            "stab_architecture": ["classic", "t_tail"]
        }
    },
    "number_of_engine": {
        "quadri": {
            "tank_architecture": ["pods"]
        },
        "hexa": {
            "tank_architecture": ["pods"]
        }
    },
    "nacelle_attachment": {
        "rear": {
            "stab_architecture": ["classic", "h_tail"]
        },
        "pods": {
            "tank_architecture": ["wing_box", "piggy_back"]
        }
    },
    "power_architecture": None,
    "power_source": {
        "fuel_cell": {
            "power_architecture": ["tf", "tp", "extf"]
        },
        "battery": {
            "power_architecture": ["tf", "tp", "extf"]
        },
        "fuel": {
            "power_architecture": ["ef", "ep", "exef"]
        }
    },
    "fuel_type": {
        "battery": {
            "power_architecture": ["tf", "tp", "extf"],
            "power_source": ["fuel", "fuel_cell"]
        },
        "kerosene": {
            "power_architecture": ["ef", "ep", "exef"],
            "power_source": ["fuel_cell", "battery"]
        },
        "methane": {
            "power_architecture": ["ef", "ep", "exef"],
            "power_source": ["fuel_cell", "battery"]
        },
        "liquid_h2": {
            "power_source": ["battery"]
        },
        "compressed_h2": {
            "power_source": ["battery"]
        }
    }
}

""""Same as INCOMPATIBILITY_DICT but in Downward order according to the order given by the ARRANGEMENT_DICT."""
INCOMPATIBILITY_DICT_DOWNWARD={
        "body_type" : {"fuselage" : {"wing_type" : ["blended"]},
                       "blended" : {"wing_type": ["classic"]}
                       },
        "wing_type" : None , # No incompatibility
        "wing_attachement": None ,
        "stab_architecture": {"classic" : {"nacelle_attachment":["rear"],
                                           "tank_architecture": ["piggy_back"]},
                              "h_tail"  : {"nacelle_attachment":["rear"]},
                              "t_tail"  : {"tank_architecture": ["piggy_back"]}
                              },
        "tank_architecture": {"pods"      : {"number_of_engine":["quadri","hexa"]},
                              "wing_box"  : {"nacelle_attachment":["pods"]},
                              "piggy_back": {"nacelle_attachment":["pods"]}
                              },
        "number_of_engine": None, # already listed in tank_architecture
        "nacelle_attachment": None, # already listed in stab_architecture and tank_architecture
        "power_architecture": {"tf"  :{"power_source":["fuel_cell","battery"],
                                       "fuel_type":["battery"]},
                               "tp"  :{"power_source":["fuel_cell","battery"],
                                       "fuel_type":["battery"]},
                               "ef"  :{"power_source":["fuel"],
                                       "fuel_type":["kerosene","methane"]},
                               "ep"  :{"power_source":["fuel"],
                                       "fuel_type":["kerosene","methane"]},
                               "pte" :{"power_source":[]},                      # TODO: is pte compatible with all power sources ?
                               "extf":{"power_source":["fuel_cell","battery"],
                                       "fuel_type":["battery"]},
                               "exef":{"power_source":["fuel"],
                                       "fuel_type":["kerosene","methane"]},
                               },
        "power_source": {"fuel"     : {"fuel_type":["battery"]},
                         "fuel_cell": {"fuel_type":["kerosene","methane","battery"]},
                         "battery"  : {"fuel_type":["kerosene","methane","liquid_h2","compressed_h2"]}
                         },
        "fuel_type": None
}


if __name__ == "__main__":
    """Convert the DOWNWARD incompatibility dict to the upward incompatibility dict"""
    new_dict = dict.fromkeys(INCOMPATIBILITY_DICT_DOWNWARD)
    for k in INCOMPATIBILITY_DICT_DOWNWARD.keys():
        if INCOMPATIBILITY_DICT_DOWNWARD[k] == None:
            continue

        for setting_key,setting_dict in INCOMPATIBILITY_DICT_DOWNWARD[k].items():
            for ikey,incompatible_settings in setting_dict.items():
                if new_dict[ikey] == None:
                    new_dict[ikey] = {} # init subdict
                for s in incompatible_settings:
                    try: new_dict[ikey][s]
                    except KeyError: new_dict[ikey][s] = {} # init subsubdict
                    try: new_dict[ikey][s][k]
                    except KeyError: new_dict[ikey][s][k] = []  # init list
                    new_dict[ikey][s][k].append(setting_key)

    # from marilib.utils.read_write import MarilibIO
    # io = MarilibIO()
    # io.to_json_file(new_dict,'temp')


class ArrangementTree(Node):
    """A custom anytree.Node object to describe all feasible arrangement"""
    def __init__(self,**kwargs):
        """ Keep only the branches of the tree passing through the selected node(s)
        :param **kwargs: the Arrangement settings that are set to a desired value. Example::
                tree = ArrangementTree(wing_type='classic')
        :return: the root node of the tree (anytree.Node)
        """
        super().__init__("Arrangement")  # intialize root node

        # build a dict of 'fixed settings' and check their validity.
        fixed_nodes = dict.fromkeys(ARRANGEMENT_DICT.keys())    # default init
        for key,val in kwargs.items():
            if key not in ARRANGEMENT_DICT.keys():
                msg = "Arrangement has no entry named %s" % key
                raise KeyError(msg)
            else:
                if val not in ARRANGEMENT_DICT[key]:
                    raise ValueError("Invalid value %s for entry %s" %(val,key))
                else:
                    fixed_nodes[key] = val

        # Construct the tree, step by step (depth level by depth level)
        for key,settings in ARRANGEMENT_DICT.items():  # iterate over all arrangement options
            if fixed_nodes[key] is not None:
                for leaf in self.leaves:  # iterate over the "leaves" of the tree
                    if self.is_feasible(self.path_of_node(leaf)+[fixed_nodes[key]]):
                        leaf.children = (Node(fixed_nodes[key]),)
                    else: # delete this branch
                        try:
                            while len(leaf.parent.children) < 2:  # find the fork at the origin of the branch
                                leaf = leaf.parent
                            leaf.parent = None  # detach the branch from the tree
                        except AttributeError:
                            print("The setting '%s' is not compatible with the other settings" %fixed_nodes[key])
                            self.root.children =[]
                            break

            else:
                for leaf in self.leaves:
                    leaf.children = (Node(s) for s in settings if len(s) > 0 and self.is_feasible(self.path_of_node(leaf)+[s]))


    def is_feasible(self,path):
        """Check that the last element of the path is compatible with all other elements in the path.
        :return: `True` if feasible, `False` if incompatible settings are found in the path
        """
        current_setting_value = path[-1]
        current_setting_key = list(INCOMPATIBILITY_DICT.keys())[len(path)-1]

        if new_dict[current_setting_key] == None:  # there is no icompatibility for this setting
            return True
        else:
            try: # try to store the dict incompatibilities of the current_setting_value.
                incompatibilities = new_dict[current_setting_key][current_setting_value] #
            except KeyError:  # the current_setting_value has no incompatibility
                incompatibilities = None
                return True

            pathdict = self.dict_from_path(path)
            for key,incomp_list in incompatibilities.items():  # test for incompatibilitites in the current path
                if pathdict[key] in incomp_list:
                    return False
            return True

        raise ValueError("Unexpected behavior, should return True are False but found None.")

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

    def path_of_node(self,node):
        """convert the builtin node.path (list of node objects) to a list of node names (list of strings)"""
        return [str(n.name) for n in node.path[1:]]

    def dict_from_path(self,path):
        """Convert a path to an arrangement dict, using the entries given by `ARRANGEMENT_DICT`"""
        dic = dict.fromkeys(ARRANGEMENT_DICT.keys())
        for key,value in zip(dic.keys(), path):
            dic[key] = value
        return dic

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
ax1.xaxis.set_visible(False)

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
        for leaf in tree.leaves:
            try:
                x,y = path_to_line(tree.path_of_node(leaf),len(cellText[0]),len(cellText))
                ax1.plot(x,y,'-k',lw=3,alpha=0.5) # 1./len(tree.leaves))
            except ValueError:
                tit.set_text("Your configuration is not feasible")
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

