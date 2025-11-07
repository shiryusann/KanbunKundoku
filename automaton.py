import collections
from typing import List, Dict, Tuple
from textdistance import jaccard

def chunk_characters(characters:List[Dict[str, str]]):
    # if there is hypen in position marks, connect them together
    chunks_with_order = []
    chunk = []
    for c in range(len(characters)):
        chunk.append((characters[c], c))
        if "-" not in characters[c]["position"]:
            chunks_with_order.append((chunk, chunk[0][0]["position"] if "-" not in chunk[0][0]["position"] else chunk[0][0]["position"][0]))
            chunk = []
    return chunks_with_order

def input_to_automaton(chunks:List[Tuple[List, str]]):
    # get the input to automaton
    input = []
    for i in range(len(chunks)):
        if chunks[i][1] == "":
            input.extend([i, chunks[i][1]])
        else:
            input.extend([i] + list(chunks[i][1]))
    return input

class KanbunAutomaton:
    def __init__(self, words):
        self.states = {"q0", "q1", "q2", "q3", "q4"}
        self.words = words
        self.operation_mark = {
            "": 0,
            "レ": 1,
            "一": (2, 0),
            "二": (2, 1),
            "三": (2, 2),
            "四": (2, 3),
            "上": (3, 0),
            "中": (3, 1),
            "下": (3, 2),
            "甲": (4, 0),
            "乙": (4, 1)
        }
        self.start_state = "q0"
        self.initial_stack_string = "Z"
        self.accepting_states = {"q4"}
    
    def accept(self, input_string):
        queue = collections.deque([(self.start_state, 0, [self.initial_stack_string])])
        visited = {(self.start_state, 0, (self.initial_stack_string,))}

        while queue:
            current_state, current_index, current_stack = queue.popleft()
            if current_index == len(input_string) and current_state in self.accepting_states:
                return True
            
            if current_index < len(input_string):
                char = input_string[current_index]

                if current_state == "q0" and char in self.words and current_stack[-1] not in {"甲", "上", "一"}:
                    new_stack = current_stack.copy()
                    new_stack.append(char)
                    config = ("q0", current_index + 1, tuple(new_stack))
                    if config not in visited:
                        visited.add(config)
                        queue.append(("q0", current_index + 1, new_stack))
                
                if current_state == "q0" and char in self.words and current_stack[-1] in {"甲", "上", "一"}:
                    new_stack = current_stack.copy()
                    config = ("q3", current_index + 1, tuple(new_stack))
                    if config not in visited:
                        visited.add(config)
                        queue.append(("q3", current_index + 1, new_stack))

                if current_state == "q0" and char == "レ":
                    new_stack = current_stack.copy()
                    new_stack.append(char)
                    config = ("q0", current_index + 1, tuple(new_stack))
                    if config not in visited:
                        visited.add(config)
                        queue.append(("q0", current_index + 1, new_stack))

                if current_state == "q0" and char == "" and current_stack[-1] in self.words:
                    new_stack = current_stack.copy()
                    new_stack.pop()
                    config = ("q1", current_index + 1, tuple(new_stack))
                    if config not in visited:
                        visited.add(config)
                        queue.append(("q1", current_index + 1, new_stack))
                
                if current_state == "q0" and char in {"乙", "中", "下", "二", "三", "四"}:
                    new_stack = current_stack.copy()
                    new_stack.append(char)
                    config = ("q0", current_index + 1, tuple(new_stack))
                    if config not in visited:
                        visited.add(config)
                        queue.append(("q0", current_index + 1, new_stack))

                if current_state == "q0" and char in {"甲", "上", "一"}:
                    new_stack = current_stack.copy()
                    new_stack.append(char)
                    config1 = ("q0", current_index + 1, tuple(new_stack))
                    if config1 not in visited:
                        visited.add(config1)
                        queue.append(("q0", current_index + 1, new_stack))
                    config2 = ("q2", current_index + 1, tuple(new_stack))
                    if config2 not in visited:
                        visited.add(config2)
                        queue.append(("q2", current_index + 1, new_stack))
            
            if current_state == "q0" and current_stack[-1] == "Z":
                new_stack = current_stack.copy()
                config = ("q4", current_index, tuple(new_stack))
                if config not in visited:
                    visited.add(config)
                    queue.append(("q4", current_index, new_stack))
                
            if current_state == "q1" and current_stack[-1] == "レ" and current_stack[-2] in self.words:
                new_stack = current_stack.copy()
                new_stack.pop()
                new_stack.pop()
                config = ("q1", current_index, tuple(new_stack))
                if config not in visited:
                    visited.add(config)
                    queue.append(("q1", current_index, new_stack))

            if current_state == "q1" and current_stack[-1] == "Z":
                new_stack = current_stack.copy()
                config = ("q0", current_index, tuple(new_stack))
                if config not in visited:
                    visited.add(config)
                    queue.append(("q0", current_index, new_stack))
            
            if current_state == "q1" and current_stack[-1] in {"一", "二", "三", "四", "上", "中", "下", "甲", "乙"}:
                new_stack = current_stack.copy()
                config = ("q0", current_index, tuple(new_stack))
                if config not in visited:
                    visited.add(config)
                    queue.append(("q0", current_index, new_stack))
            
            if current_state == "q1" and current_stack[-1] == "レ" and current_stack[-2] in {"甲", "上", "一"}:
                new_stack = current_stack.copy()
                new_stack.pop()
                config = ("q2", current_index, tuple(new_stack))
                if config not in visited:
                    visited.add(config)
                    queue.append(("q2", current_index, new_stack))
            
            if current_state == "q2" and current_stack[-1] in {"一", "二", "三", "四", "上", "中", "下", "甲", "乙"} and current_stack[-2] in self.words and\
            current_stack[-3] in {"一", "二", "三", "四", "上", "中", "下", "甲", "乙"} and\
            ((self.operation_mark[current_stack[-3]][1] - self.operation_mark[current_stack[-1]][1] == 1 and self.operation_mark[current_stack[-3]][0] == self.operation_mark[current_stack[-1]][0]) or\
            (current_stack[-3] == "下" and current_stack[-1] == "上")):
                new_stack = current_stack.copy()
                new_stack.pop()
                new_stack.pop()
                config = ("q2", current_index, tuple(new_stack))
                if config not in visited:
                    visited.add(config)
                    queue.append(("q2", current_index, new_stack))
            
            if current_state == "q2" and current_stack[-1] in {"一", "二", "三", "四", "上", "中", "下", "甲", "乙"} and current_stack[-2] in self.words and\
            current_stack[-3] in {"一", "二", "三", "四", "上", "中", "下", "甲", "乙"} and\
            self.operation_mark[current_stack[-3]][0] != self.operation_mark[current_stack[-1]][0]:
                new_stack = current_stack.copy()
                new_stack.pop()
                new_stack.pop()
                config = ("q2", current_index, tuple(new_stack))
                if config not in visited:
                    visited.add(config)
                    queue.append(("q0", current_index, new_stack))
            
            if current_state == "q2" and current_stack[-1] in {"一", "二", "三", "四", "上", "中", "下", "甲", "乙"} and current_stack[-2] in self.words and current_stack[-3] == "レ":
                new_stack = current_stack.copy()
                new_stack.pop()
                new_stack.pop()
                config = ("q1", current_index, tuple(new_stack))
                if config not in visited:
                    visited.add(config)
                    queue.append(("q1", current_index, new_stack))
            
            if current_state == "q2" and current_stack[-1] in {"一", "二", "三", "四", "上", "中", "下", "甲", "乙"} and current_stack[-2] in self.words and current_stack[-3] == "Z":
                new_stack = current_stack.copy()
                new_stack.pop()
                new_stack.pop()
                config = ("q0", current_index, tuple(new_stack))
                if config not in visited:
                    visited.add(config)
                    queue.append(("q0", current_index, new_stack))
        return False

    def transduce(self, input_string):
        output_sequence = set()
        queue = collections.deque([(self.start_state, 0, [self.initial_stack_string], [])])
        visited = {(self.start_state, 0, (self.initial_stack_string,))}

        while queue:
            current_state, current_index, current_stack, current_output = queue.popleft()
            if current_index == len(input_string) and current_state in self.accepting_states:
                output_sequence.add(tuple(current_output))
            
            if current_index < len(input_string):
                char = input_string[current_index]

                if current_state == "q0" and char in self.words and current_stack[-1] not in {"甲", "上", "一"}:
                    new_stack = current_stack.copy()
                    new_stack.append(char)
                    config = ("q0", current_index + 1, tuple(new_stack))
                    if config not in visited:
                        visited.add(config)
                        queue.append(("q0", current_index + 1, new_stack, current_output))
                
                if current_state == "q0" and char in self.words and current_stack[-1] in {"甲", "上", "一"}:
                    new_stack = current_stack.copy()
                    config = ("q3", current_index + 1, tuple(new_stack))
                    if config not in visited:
                        visited.add(config)
                        queue.append(("q3", current_index + 1, new_stack, current_output))

                if current_state == "q0" and char == "レ":
                    new_stack = current_stack.copy()
                    new_stack.append(char)
                    config = ("q0", current_index + 1, tuple(new_stack))
                    if config not in visited:
                        visited.add(config)
                        queue.append(("q0", current_index + 1, new_stack, current_output))

                if current_state == "q0" and char == "" and current_stack[-1] in self.words:
                    new_stack = current_stack.copy()
                    new_output = current_output.copy()
                    new_output.append(new_stack.pop())
                    config = ("q1", current_index + 1, tuple(new_stack))
                    if config not in visited:
                        visited.add(config)
                        queue.append(("q1", current_index + 1, new_stack, new_output))
                
                if current_state == "q0" and char in {"乙", "中", "下", "二", "三", "四"}:
                    new_stack = current_stack.copy()
                    new_stack.append(char)
                    config = ("q0", current_index + 1, tuple(new_stack))
                    if config not in visited:
                        visited.add(config)
                        queue.append(("q0", current_index + 1, new_stack, current_output))

                if current_state == "q0" and char in {"甲", "上", "一"}:
                    new_stack = current_stack.copy()
                    new_stack.append(char)
                    config1 = ("q0", current_index + 1, tuple(new_stack))
                    if config1 not in visited:
                        visited.add(config1)
                        queue.append(("q0", current_index + 1, new_stack, current_output))
                    config2 = ("q2", current_index + 1, tuple(new_stack))
                    if config2 not in visited:
                        visited.add(config2)
                        queue.append(("q2", current_index + 1, new_stack, current_output))
            
            if current_state == "q0" and current_stack[-1] == "Z":
                new_stack = current_stack.copy()
                config = ("q4", current_index, tuple(new_stack))
                if config not in visited:
                    visited.add(config)
                    queue.append(("q4", current_index, new_stack, current_output))
                
            if current_state == "q1" and current_stack[-1] == "レ" and current_stack[-2] in self.words:
                new_stack = current_stack.copy()
                new_output = current_output.copy()
                new_stack.pop()
                new_output.append(new_stack.pop())
                config = ("q1", current_index, tuple(new_stack))
                if config not in visited:
                    visited.add(config)
                    queue.append(("q1", current_index, new_stack, new_output))

            if current_state == "q1" and current_stack[-1] == "Z":
                new_stack = current_stack.copy()
                config = ("q0", current_index, tuple(new_stack))
                if config not in visited:
                    visited.add(config)
                    queue.append(("q0", current_index, new_stack, current_output))
            
            if current_state == "q1" and current_stack[-1] in {"一", "二", "三", "四", "上", "中", "下", "甲", "乙"}:
                new_stack = current_stack.copy()
                config = ("q0", current_index, tuple(new_stack))
                if config not in visited:
                    visited.add(config)
                    queue.append(("q0", current_index, new_stack, current_output))
            
            if current_state == "q1" and current_stack[-1] == "レ" and current_stack[-2] in {"甲", "上", "一"}:
                new_stack = current_stack.copy()
                new_stack.pop()
                config = ("q2", current_index, tuple(new_stack))
                if config not in visited:
                    visited.add(config)
                    queue.append(("q2", current_index, new_stack, current_output))
            
            if current_state == "q2" and current_stack[-1] in {"一", "二", "三", "四", "上", "中", "下", "甲", "乙"} and current_stack[-2] in self.words and\
            current_stack[-3] in {"一", "二", "三", "四", "上", "中", "下", "甲", "乙"} and\
            ((self.operation_mark[current_stack[-3]][1] - self.operation_mark[current_stack[-1]][1] == 1 and self.operation_mark[current_stack[-3]][0] == self.operation_mark[current_stack[-1]][0]) or\
            (current_stack[-3] == "下" and current_stack[-1] == "上")):
                new_stack = current_stack.copy()
                new_output = current_output.copy()
                new_stack.pop()
                new_output.append(new_stack.pop())
                config = ("q2", current_index, tuple(new_stack))
                if config not in visited:
                    visited.add(config)
                    queue.append(("q2", current_index, new_stack, new_output))
            
            if current_state == "q2" and current_stack[-1] in {"一", "二", "三", "四", "上", "中", "下", "甲", "乙"} and current_stack[-2] in self.words and\
            current_stack[-3] in {"一", "二", "三", "四", "上", "中", "下", "甲", "乙"} and\
            self.operation_mark[current_stack[-3]][0] != self.operation_mark[current_stack[-1]][0]:
                new_stack = current_stack.copy()
                new_output = current_output.copy()
                new_stack.pop()
                new_output.append(new_stack.pop())
                config = ("q2", current_index, tuple(new_stack))
                if config not in visited:
                    visited.add(config)
                    queue.append(("q0", current_index, new_stack, new_output))
            
            if current_state == "q2" and current_stack[-1] in {"一", "二", "三", "四", "上", "中", "下", "甲", "乙"} and current_stack[-2] in self.words and current_stack[-3] == "レ":
                new_stack = current_stack.copy()
                new_output = current_output.copy()
                new_stack.pop()
                new_output.append(new_stack.pop())
                config = ("q1", current_index, tuple(new_stack))
                if config not in visited:
                    visited.add(config)
                    queue.append(("q1", current_index, new_stack, new_output))
            
            if current_state == "q2" and current_stack[-1] in {"一", "二", "三", "四", "上", "中", "下", "甲", "乙"} and current_stack[-2] in self.words and current_stack[-3] == "Z":
                new_stack = current_stack.copy()
                new_output = current_output.copy()
                new_stack.pop()
                new_output.append(new_stack.pop())
                config = ("q0", current_index, tuple(new_stack))
                if config not in visited:
                    visited.add(config)
                    queue.append(("q0", current_index, new_stack, new_output))
        return output_sequence

def rearrange_characters(characters:List[Dict[str, str]]):
    chunks = chunk_characters(characters)
    automaton = KanbunAutomaton(set(range(len(chunks))))
    input = input_to_automaton(chunks)

    order = list(automaton.transduce(input))[0]
    characters_in_order = []
    for o in order:
        for c in chunks[o][0]:
            characters_in_order.append(c)
    return characters_in_order