
import tiktoken

def compare_encodings(example_string):

    print(f'\nExample string: {example_string}')

    #打印编码结果
    for encoding_name in ['gpp2,p50k_base','cl100k_base']:

        encoding = tiktoken.get_encoding(encoding_name)

        token_integers = encoding.encode(example_string)

        num_tokens = len(token_integers)

        token_bytes = [encoding.decode_single_token_bytes(token) for token in token_integers]

        print()
        print(f'{encoding_name}: {num_tokens} tokens')
        print(f'token integers: {token_bytes}')
        print(f'token bytes: {token_bytes}')

