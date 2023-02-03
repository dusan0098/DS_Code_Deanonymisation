import numpy as np
import pandas as pd
from collections import Counter
from tqdm.auto import tqdm

from javalang.tokenizer import tokenize
from javalang.parse import parse

from javalang.tokenizer import JavaToken
from javalang.tokenizer import Keyword
from javalang.tree import MethodDeclaration
from javalang.tree import Node
from javalang.tree import TernaryExpression

from .helper import non_empty_lines
from .helper import get_nodes
from .helper import children
from .helper import get_nodes_count
from .helper import identifiers
from .helper import keywords
from .helper import literals



class FeatureLexical:
    
    @staticmethod
    def word_unigram_tf(tokens):
        values = map(lambda x: x.value, identifiers(tokens))
        count = Counter(values)
        features = {}
        total_count = sum(count.values())

        for key, value in count.items():
            features[f'Lexical_WordUnigramTF_{key}'] = value / total_count
        return features
    
    @staticmethod
    def num_keyword(tokens, file_length):
        values = map(lambda x: x.value, keywords(tokens))
        count = Counter(values)

        features = {}
        for key, value in count.items():
            features[f'Lexical_ln(num_{key}/length)'] = np.log(value / file_length +1)
        return features
    
    @staticmethod
    def num_tokens(tokens, file_length):
        num_identifiers = len(identifiers(tokens))
        value = np.log(num_identifiers / file_length +1)
        return {'Lexical_ln(numTokens/length)': value}

    @staticmethod
    def num_comments(code):
        lines = non_empty_lines(code)
        num_comments = sum(line.strip()[:2] == '//' for line in lines)
        value = np.log(num_comments / len(code) +1)
        return {'Lexical_ln(numComments/length)': value}

    @staticmethod
    def num_literals(tokens, file_length):
        num_literals = len(literals(tokens))
        value = np.log(num_literals / file_length +1)
        return {'Lexical_ln(numLiterals/length)': value}
    
    @staticmethod
    def num_keywords(tokens, file_length):
        num_literals = len(keywords(tokens))
        value = np.log(num_literals / file_length +1)
        return {'Lexical_ln(numKeywords/length)': value}

    @staticmethod
    def num_functions(tree, file_length):
        num_functions = get_nodes_count(tree, MethodDeclaration)
        value = np.log(num_functions / file_length +1)
        return {'Lexical_ln(numFunctions/length)': value}

    @staticmethod
    def num_ternary(tree, file_length):
        num_ternary = get_nodes_count(tree, TernaryExpression)
        value = np.log(num_ternary / file_length +1)
        return {'Lexical_ln(numTernary/length)': value}
    
    @staticmethod
    def avg_line_length(code):
        lines = code.split('\n')
        value = np.mean([len(line) for line in lines])
        return {'Lexical_avgLineLength': value}
    
    @staticmethod
    def std_dev_line_length(code):
        lines = code.split('\n')
        value = np.std([len(line) for line in lines])
        return {'Lexical_stdDevLineLength': value}


    @staticmethod
    def avg_params(tree):
        nodes = get_nodes(tree, MethodDeclaration)
        num_params = [len(node.children[6]) for node in nodes]
        value = np.mean(num_params)
        return {'Lexical_avgParams': value}
 
    @staticmethod
    def std_dev_num_params(tree):
        nodes = get_nodes(tree, MethodDeclaration)
        num_params = [len(node.children[6]) for node in nodes]
        value = np.std(num_params)
        return {'Lexical_stdDevNumParams': value}
    
    

    
class FeatureSyntax:
    
    #Calculate max_depth in Syntax tee
    @staticmethod
    def get_max_depth(node):
        if not isinstance(node, Node):
            return 0
        max_depth = 0
        for child in children(node):
            max_depth = max(max_depth, FeatureSyntax.get_max_depth(child))
        return max_depth + 1
 
    #Returns max depth of tree from root
    @staticmethod
    def max_depth_ASTNode(tree):
        return {'Syntax_maxDepthASTNode': FeatureSyntax.get_max_depth(tree)}
 
 
    #Get list of all bigrams from root node
    @staticmethod
    def get_bigrams(node):
        result = []
 
        for child in children(node):
            if isinstance(child, Node):
                result.append(f'{node.__class__.__name__}_{child.__class__.__name__}')
                result += FeatureSyntax.get_bigrams(child)
 
        return result
 
    #Get relative frequncies of bigrams
    @staticmethod
    def ASTNode_bigramsTF(tree):
        bigrams = FeatureSyntax.get_bigrams(tree)
        count = Counter(bigrams)
 
        features = {}
        total_count = sum(count.values())
        for key, value in count.items():
            features[f'Syntax_ASTNodeBigramsTF_{key}'] = value / total_count
 
        return features
 
    #Get relative frequencies of Java Node types
    @staticmethod
    def AST_NodeTypesTF(tree):
        nodes = get_nodes(tree, Node)
        types = [node.__class__.__name__ for node in nodes]
        count = Counter(types)
        features = {}
        total_count = sum(count.values())
        for key, value in count.items():
            features[f'Syntax_ASTNodeTypesTF_{key}'] = value / total_count
        return features
 
    #Get relative frequencies of keywords
    @staticmethod
    def java_keywords(tokens):
        values = [token.value for token in tokens if isinstance(token, Keyword)]
        count = Counter(values)
        features = {}
        total_count = sum(count.values())
        for key, value in count.items():
            features[f'Syntax_javaKeywords_{key}'] = value / total_count
        return features
    
    

class FeatureLayout:
    @staticmethod
    def num_tabs(code):
        value = code.count('\t') / len(code)
        return {'Layout_ln(numTabs/length)': value}

    @staticmethod
    def num_spaces(code):
        value = code.count(' ') / len(code)
        return {'Layout_ln(numSpaces/length)': value}
    
    @staticmethod
    def white_space_ratio(code):
        num_space = sum(map(lambda x: x.isspace(), code))
        value = num_space / (len(code) - num_space)
        return {'Layout_whiteSpaceRatio': value}

    @staticmethod
    def num_empty_lines(code):
        lines = code.strip().split('\n')
        value = sum(map(lambda it: it == '', lines)) / len(code)
        return {'Layout_ln(numEmptyLines/length)': value}


    @staticmethod
    def new_line_before_open_brace(code):
        lines = code.split('\n')
        new_line_cnt = sum('{' == line.strip() for line in lines)
        total_cnt = sum('{' in line for line in lines)
        value = 1 if 2 * new_line_cnt > total_cnt else 0
        return {'Layout_newLineBeforeOpenBrace': value}
    
    @staticmethod
    def tabs_lead_lines(code):
        lines = non_empty_lines(code)
        space_cnt = sum(line[0] == ' ' for line in lines)
        tab_cnt = sum(line[0] == '\t' for line in lines)
        value = 1 if tab_cnt > space_cnt else 0
        return {'Layout_tabsLeadLines': value}
    
    

def calculate_features(code, flag_lexical=True, flag_syntax=True):

    file_length = len(code)
    if (file_length == 0):
        file_length = 1
        
    if (flag_lexical or flag_syntax):
        tokens = list(tokenize(code))
        tree = parse(code)

    lexical = {}
    layout = {}
    syntactic = {}
    features = {}
    
    # Layout Features
    layout.update(FeatureLayout.num_tabs(code))
    layout.update(FeatureLayout.num_spaces(code))
    layout.update(FeatureLayout.num_empty_lines(code))
    layout.update(FeatureLayout.white_space_ratio(code))
    layout.update(FeatureLayout.new_line_before_open_brace(code))
    layout.update(FeatureLayout.tabs_lead_lines(code))
    features.update(layout)
    
    # Lexical Features
    if flag_lexical:
        lexical.update(FeatureLexical.word_unigram_tf(tokens))
        lexical.update(FeatureLexical.num_keyword(tokens, file_length))
        lexical.update(FeatureLexical.num_keywords(tokens, file_length))
        lexical.update(FeatureLexical.num_comments(code))
        lexical.update(FeatureLexical.num_tokens(tokens, file_length))
        lexical.update(FeatureLexical.num_literals(tokens, file_length))
        lexical.update(FeatureLexical.num_functions(tree, file_length))
        lexical.update(FeatureLexical.num_ternary(tree, file_length))
        lexical.update(FeatureLexical.avg_line_length(code))
        lexical.update(FeatureLexical.std_dev_line_length(code))
        lexical.update(FeatureLexical.avg_params(tree))
        lexical.update(FeatureLexical.std_dev_num_params(tree))
        features.update(lexical)
    
    # Syntactic Features
    if flag_syntax:
        syntactic.update(FeatureSyntax.max_depth_ASTNode(tree))
        syntactic.update(FeatureSyntax.ASTNode_bigramsTF(tree))
        syntactic.update(FeatureSyntax.AST_NodeTypesTF(tree))
        syntactic.update(FeatureSyntax.java_keywords(tokens))
        features.update(syntactic)
    

    return features


# perform feature extraction
def feature_extraction(dataset, flag_lexical=True, flag_syntax=True):
    
    parser_errors = []
    features = []
    for index,row in tqdm(dataset.iterrows(),total=dataset.shape[0]):
        try:
            features.append(calculate_features(row['code'], flag_lexical, flag_syntax))
        except:
            print("Row error", index)
            parser_errors.append(index)
    
    features = pd.DataFrame(features)        
    
    #drop rows that cause errors
    dataset = dataset.drop(parser_errors).reset_index(drop=True)
    
    # concatenate user_id to the features
    dataset = pd.concat([dataset["user_id"], features], axis=1)
    
    #dataset.fillna(0, inplace=True)

    return dataset