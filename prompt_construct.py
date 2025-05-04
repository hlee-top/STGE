def get_ner_prompt(data, shot_data=None, template=None):
    ent_type = ", ".join(template["entity_type"])
    if shot_data is None or len(shot_data) == 0:
        demo = ""
    else:
        demo_list = []
        for data_ex in shot_data:
            if data_ex["outputs"] == "{}":
                outputs = " { }"
            else:
                outputs = data_ex["outputs"].replace('"', ' " ').replace("]", "] ").replace('  "', ' "')
            demo_list.append("Context: {}\nEntity type: {}\nAnswer: {} ".format(
                data_ex["inputs"], ent_type, outputs))
        demo = "The format refers to the demonstration below:\n{} \n".format("\n".join(demo_list))
    context = data["inputs"]

    return ("Entity is words with specific meanings in the context, mainly including names of people, places, "
            "institutions, proper nouns, etc. Please output the entities in the following context. The format of "
            "the output is json, the key is the entity type, and the value is an entity list. Entity must be consecutive "
            "words in the context. If the entity type does not have a corresponding entity, the key and value will "
            "not be output.\n{}Context: {}\nEntity type: {}\nAnswer: ").format(demo, context, ent_type)


def get_re_prompt(data, shot_data=None, template=None):
    re_type = ", ".join(template["relation_type"])
    if shot_data is None or len(shot_data) == 0:
        demo = ""
    else:
        demo_list = []
        for data_ex in shot_data:
            if data_ex["outputs"] == "{}":
                outputs = " { }"
            else:
                outputs = data_ex["outputs"].replace('"', ' " ').replace("[[", "[ [").replace("]", "] ").replace('  "',
                                                                                                                 ' "')
            demo_list.append("Context: {}\nRelation type: {}\nAnswer: {} ".format(
                data_ex["inputs"], re_type, outputs))
        demo = "The format refers to the demonstration below:\n{} \n".format("\n".join(demo_list))
    context = data["inputs"]
    return ('The goal of relation extraction is to extract entities with specific relation types '
            'from the context. Please output the relation tuples in the following context. The format of the output '
            'is json, the key is the relation type, the value is a list, each element in the list is a list of '
            'length 2 consisting of entity1 and entity2. Entity must be consecutive words in the context. '
            'If the relation type does not have a corresponding entity, the key and value will '
            'not be output.\n{}Context: {}\nRelation type: {}\nAnswer: '.format(demo, context, re_type))


def get_eae_prompt(data, shot_data=None, event2argument=None):
    if shot_data is None or len(shot_data) == 0:
        demo = ""
    else:
        demo_list = []
        for data_ex in shot_data:
            trigger_str = "Trigger word: {}".format(data_ex["gold_dict"]["trigger"]["text"])
            if data_ex["outputs"] == "{}":
                outputs = " { }"
            else:
                outputs = data_ex["outputs"].replace('"', ' " ').replace("]", "] ").replace('  "', ' "')
            demo_list.append("Context: {}\n{}\nEvent type: {}\nRole: {}\nAnswer: {} ".format(
                data_ex["inputs"], trigger_str, data_ex["gold_dict"]["event_type"],
                ", ".join(event2argument[data_ex["gold_dict"]["event_type"]]["role_list"]),
                outputs))
        demo = "The format refers to the demonstration below:\n{} \n".format("\n".join(demo_list))
    event_type = data["gold_dict"]["event_type"]
    trigger_word = data["gold_dict"]["trigger"]["text"]
    context = data["inputs"]
    role = ", ".join(event2argument[event_type]["role_list"])
    trigger_str = "Trigger word: {}".format(trigger_word)


    return "Trigger words represent the words that best represent the occurrence of an event, " \
               "usually a verb or noun. Event arguments are entities that play specific roles in events. The " \
               "following context represents a {} event triggered by {}. The format of the output is json, " \
               "the key is the specific role played by the argument, and the value is an argument list. " \
               "Argument must be consecutive words in the context. " \
               "If the argument role does not have a corresponding argument, the key and value will not be output.\n" \
               "{}Context: {}\n{}\nEvent type: {}\nRole: {}\nAnswer: ".format(event_type, trigger_word, demo,
                                                                       context, trigger_str, event_type, role)

def get_response(output, split_str="Answer: ", format_symbols=None):
    output = output.split(split_str)[-1].strip()
    return output
