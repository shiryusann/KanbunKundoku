import re
import json
from typing import List, Tuple, Dict
from automaton import chunk_characters, input_to_automaton, KanbunAutomaton, rearrange_characters

def process_label(characters:List[Dict], characters_in_order:List[Tuple[Dict, int]]):
    for c in range(len(characters_in_order)):
        if "B" in characters_in_order[c][0]["particle"]:
            index = characters_in_order[c - 1][1]
            back_append = re.search(r"B[^S]*", characters_in_order[c][0]["particle"]).group(0)
            characters[characters_in_order[c][1]]["particle"] = re.sub(r"B[^S]*", "", characters[characters_in_order[c][1]]["particle"])
            characters[index]["particle"] += back_append.replace("B", "")
    return characters

def final_mark(characters:List[Dict]):
    final_characters = []
    for c in characters:
        final_character = {}
        final_character["character"] = c["character"]
        final_character["right"] = c["okurigana"]
        final_character["left"] = c["position"]
        particle = c["particle"]
        if "D" in c["particle"]:
            if "ず" in c["particle"]:
                particle = particle[2:]
            else:
                particle = re.sub(r"D[^K]*", "", particle)
        if "K" in c["particle"]:
            if "S" in c["particle"]:
                particle = re.sub(r"K.*", "", particle)
            else:
                particle = particle.replace("K", "")
        if "S" in c["particle"]:
            particle = re.sub(r"S[^K]*", "", particle)
        if final_character["right"] != particle:
            final_character["right"] += particle
        final_characters.append(final_character)
    return final_characters

def find_last_number(numbers:List[int], target:int):
    n = -1
    for i in numbers:
        if i < target:
            n = i
    return n

def translate(characters_in_order:List[Tuple[Dict, int]]):
    translated_sentence = []
    for c in range(len(characters_in_order)):
        translated_segment = ""
        if "D" in characters_in_order[c][0]["particle"]:
            translated_segment += characters_in_order[c][0]["okurigana"] + characters_in_order[c][0]["particle"].replace("D", "").replace("B", "").replace("K", "")
        elif "S" in characters_in_order[c][0]["particle"]:
            translated_segment += characters_in_order[c][0]["okurigana"] + re.search(r"S.*$", characters_in_order[c][0]["particle"]).group(0).replace("S", "").replace("K", "")
            if "B" in characters_in_order[c][0]["particle"]:
                translated_sentence[c - 1] +=  re.search(r"B[^S]*", characters_in_order[c][0]["particle"]).group(0).replace("B", "")
            characters_before = [characters_in_order[i][1] for i in range(c)]
            character_number_before = find_last_number(characters_before, characters_in_order[c][1])
            if character_number_before == -1:
                translated_sentence[0] = characters_in_order[c][0]["character"] + re.search(r"^[^BS]*", characters_in_order[c][0]["particle"]).group(0) + translated_sentence[0]
            else:
                translated_sentence[characters_before.index(character_number_before)] += characters_in_order[c][0]["character"] + re.search(r"^[^BS]*", characters_in_order[c][0]["particle"]).group(0)
        else:
            translated_segment += characters_in_order[c][0]["character"] + characters_in_order[c][0]["okurigana"] + characters_in_order[c][0]["particle"]
        translated_sentence.append(translated_segment)
    return "".join(translated_sentence)

def valid_marks(path:str, characters:List[Dict]):
    valid_count = 0
    original_marks = []
    valid_marks = []

    with open(path, "r", encoding = "utf-8") as file:
        sentences = json.load(file)

    for i in range(len(characters)):
        chunks = chunk_characters(characters[i]["characters"])
        automaton = KanbunAutomaton(set(range(len(chunks))))
        input = input_to_automaton(chunks)

        if not automaton.accept(input):
            continue

        valid_count += 1
        valid_marks.append(characters[i])
        original_marks.append(sentences[i])

    return original_marks, valid_marks, valid_count

def translate_corpus(original_marks:List[Dict], valid_marks:List[Dict]):
    valid_count = 0
    original_sentences = []
    translated_sentences = []
    
    for i in range(len(valid_marks)):
        try:
            translated_sentence = translate(rearrange_characters(valid_marks[i]["characters"]))
        except Exception:
            continue
        
        valid_count += 1
        translated_sentences.append(translated_sentence)
        original_sentences.append(original_marks[i]["japanese_translation"])
    
    return original_sentences, translated_sentences, valid_count