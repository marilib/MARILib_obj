from marilib.aircraft import Arrangement
import matplotlib.pyplot as plt
from anytree import Node, RenderTree, AsciiStyle, LevelOrderIter

agmt = Arrangement(body_type = "fuselage",          # "fuselage" or "blended"
                   wing_type = "classic",           # "classic" or "blended"
                   wing_attachment = "low",         # "low" or "high"
                   stab_architecture = "classic",   # "classic", "t_tail" or "h_tail"
                   tank_architecture = "pods",      # "wing_box", "piggy_back" or "pods"
                   number_of_engine = "twin",       # "twin" or "quadri"
                   nacelle_attachment = "pods",     # "wing", "rear" or "pods"
                   power_architecture = "tf",       # "tf", "extf", "ef", "exef", "tp", "ep"
                   power_source = "fuel",           # "fuel", "battery", "fuel_cell"
                   fuel_type = "liquid_h2")         # "kerosene", "methane", "liquid_h2", "Compressed_h2", "battery"

ARRANGEMENT_DICT={
          "body_type" :           ["fuselage", "blended"   , ""         , ""             , ""       , ""  ],
          "wing_type" :           ["classic" , "blended"   , ""         , ""             , ""       , ""  ],
          "wing_attachment":      ["low"     , "high"      , ""         , ""             , ""       , ""  ],
          "number_of_engine":     ["twin"    , "quadri"    , ""         , ""             , ""       , ""  ],
          "stab_architecture":    ["classic" , "t_tail"    , "h_tail"   , ""             , ""       , ""  ],
          "tank_architecture":    ["wong_box", "piggy_back", "pods"     , ""             , ""       , ""  ],
          "number_of_engine":     ["twin"    , "quadri"    , ""         , ""             , ""       , ""  ],
          "nacelle_atttachment" : ["wing"    , "rear"      , "pods"     , ""             , ""       , ""  ],
          "power_architecture":   ["tf"      , "extf"      , "ef"       , "exef"         , "tp"     , "ep"],
          "power_source" :        ["fuel"    , "fuel_cell" , "battery"  , ""             , ""       , ""  ],
          "fuel_type":            ["kerosene", "methane"   , "liquid_h2", "Compressed_h2", "battery", ""  ]
          }

def build_tree(selected_nodes):
    """ Keep only the branches of the tree passing through the selected node(s)
    :param selected_node: the list of nodes used for filtering.
    :return: the root node of the tree (anytree.Node)
    """
    # Build the tree of feasible Arrangement choices
    feasible_settings = Node("Arrangement")  # root node
    for settings in ARRANGEMENT_DICT.values():  # iterate over all options
        for leaf in feasible_settings.leaves:  # iterate over the "leaves" of the tree
            leaf.children = (Node(setting) for setting in settings if (len(setting)>0 and setting is not "blended"))

def check_feasible(node):
    """
    TODO : filter for feasible arrangement choices
    :param node: the root node of the tree (type anytree.Node)
    :return:
    """
    return None

def savetext(filename,root_node):
    """Save the tree to text file
    :param root_node:
    :return:
    """
    with open(filname,"w") as f:
        content = ""
        for pre,_, node in RenderTree(root_node,style = AsciiStyle):
            content += "%s%s\n" %(pre, node.name)
        f.write(content)







#-------------------------------------------------
# Plot the table of Arrangement settings
#-------------------------------------------------

colLabels = [k for k in ARRANGEMENT_DICT.keys()]
cellText = [[p for p in options] for options in ARRANGEMENT_DICT.values()]
cellText = list(zip(*cellText))  # trick to transpose the 2D list

fig = plt.figure("Arrangement options", figsize=(15,5))
tab = plt.table(colLabels=colLabels, colColours=['g']*len(colLabels),
                cellText = cellText,rowLoc='center', cellLoc='center',
                bbox=[0,0,1,1])
tab.auto_set_font_size(False)
tab.set_fontsize = 12
for k, cell in tab._cells.items():
    cell.set_edgecolor("silver")

def onclick(event):
    ix, iy = event.xdata, event.ydata
    if tab.contains(event)[0]:
        for k,c in tab.get_celld().items(): # find the new selected cell
            if c.contains(event)[0]:
                c.set_facecolor('r')
                print(c.get_text())
        plt.draw()

fig.canvas.mpl_connect('button_press_event', onclick)

plt.axis("on")
plt.show()



