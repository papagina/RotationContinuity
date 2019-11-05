
import re
from collections import OrderedDict



def identifier(scanner, token):  return "IDENT", token
def operator(scanner, token):    return "OPERATOR", token
def digit(scanner, token):       return "DIGIT", token
def open_brace(scanner, token):  return "OPEN_BRACE", token
def close_brace(scanner, token): return "CLOSE_BRACE", token



def new_bone(parent, name):
    bone = { "parent" : parent, "channels" : [], "offsets" : []}
    return bone

def push_bone_context(name, bone_context):
    
    bone_context.append(name)
    return bone_context

def get_bone_context(bone_context):
    return bone_context[len(bone_context)-1]

def pop_bone_context(bone_context):
    bone_context = bone_context[:-1]
    return bone_context#[len(bone_context)-1]

reserved      = [ "HIERARCHY", "ROOT", "OFFSET", "CHANNELS", "MOTION" ]
channel_names = [ "Xposition", "Yposition", "Zposition",  "Zrotation", "Xrotation",  "Yrotation" ]

scanner = re.Scanner([
    (r"[a-zA-Z_]\w*", identifier), (r"-*[0-9]+(\.[0-9]+)?", digit),
    (r"/-?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?/", digit),
    (r"}", close_brace),
    (r"{", open_brace),
    (r":", None),
    
    (r"\s+", None),
    ])

def read_offset(bvh, token_index):
    if (bvh[token_index] != ("IDENT", "OFFSET")):
        return None, None
    token_index = token_index + 1
    offsets = [ 0.0 ] * 3
    for i in range(0,3):
        #print(bvh[token_index])
        #print(float(bvh[token_index][1]))
        offsets[i] = float(bvh[token_index][1])
        token_index = token_index + 1
    return  offsets, token_index

def read_channels(bvh, token_index):
    if (bvh[token_index] != ("IDENT", "CHANNELS")):
        return None, None
    token_index = token_index + 1
    channel_count = int(bvh[token_index][1])
    token_index = token_index+1
    channels = [ "" ] * channel_count
    for i in range(0, channel_count):
        channels[i] = bvh[token_index][1]
        token_index = token_index+1
    return channels, token_index

def parse_joint(bvh, token_index, skeleton, bone_context, non_end_bones, motion_channels):
    #print bone_context
    end_site = False
    joint_id = bvh[token_index][1]
    token_index = token_index + 1
    joint_name = bvh[token_index][1]
    token_index = token_index + 1
    if (joint_id == "End"): # end site
        joint_name = get_bone_context(bone_context) + "_Nub"
        end_site = True
    joint = new_bone(get_bone_context(bone_context), joint_name)
    if bvh[token_index][0] != "OPEN_BRACE":
        print ("Was expecting brace, got ", bvh[token_index])
        return None
    #print bone_context
    token_index = token_index + 1
    offsets, token_index = read_offset(bvh, token_index)
    joint["offsets"] = offsets
    if (not(end_site)):
        channels, token_index = read_channels(bvh, token_index)
        joint["channels"] = channels
        non_end_bones.append(joint_name)
        for channel in channels:
            motion_channels.append((joint_name, channel))
    skeleton[joint_name] = joint
    #print bone_context
    while (((bvh[token_index][0] == "IDENT") and (bvh[token_index][1] == "JOINT")) or
          ((bvh[token_index][0] == "IDENT") and (bvh[token_index][1] == "End"))):
        #print "before",bone_context
        bone_context=push_bone_context(joint_name, bone_context)
        #print "after",bone_context
        (token_index, skeleton, bone_context,non_end_bones, motion_channels) = parse_joint(bvh,  token_index, skeleton, bone_context, non_end_bones, motion_channels)
        bone_context=pop_bone_context(bone_context)
        #print "pop", bone_context
    if (bvh[token_index][0]) == "CLOSE_BRACE":
        return token_index + 1, skeleton, bone_context,non_end_bones, motion_channels
    print ("Unexpected token ", bvh[token_index])

def parse_hierarchy(bvh):
    #global current_token
    current_token = 0
    if (bvh[current_token] != ("IDENT", "HIERARCHY")):
        return None
    current_token = current_token + 1
    if (bvh[current_token] != ("IDENT", "ROOT")):
        return None
    current_token = current_token + 1
    if (bvh[current_token][0] != "IDENT"):
        return None
    root_name =bvh[current_token][1]
    root_bone = new_bone(None, root_name)
    current_token = current_token + 1
    if bvh[current_token][0] != "OPEN_BRACE":
        return None
    current_token = current_token + 1
    offsets, current_token = read_offset(bvh, current_token)
    channels, current_token = read_channels(bvh, current_token)
    root_bone["offsets"]  = offsets
    root_bone["channels"] = channels
    skeleton=OrderedDict()
    skeleton[root_name] = root_bone
    bone_context = []
    motion_channels=[]
    non_end_bones=[]
    bone_context=push_bone_context(root_name,bone_context)
    
    # print "Root ", root_bone
    while(bvh[current_token][1] == "JOINT"):
        ( current_token, skeleton, bone_context, non_end_bones, motion_channels) = parse_joint(bvh,  current_token, skeleton, bone_context, non_end_bones, motion_channels)
    return  skeleton, non_end_bones



def read_bvh_hierarchy(bvh_filename):
	bvh_file = open(bvh_filename, "r")
	bvh = bvh_file.read()
	bvh_file.close()
	tokens, remainder = scanner.scan(bvh)
	skeleton = OrderedDict()
	skeleton, non_end_bones=parse_hierarchy(tokens)
	return skeleton, non_end_bones
