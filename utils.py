import pdb

def preprocess_token(token):
	token = token.lower()

	if len(token) == 1:
		if token.isalnum() or token.isascii():
			return token
		return 'unk'
		
	_token = ''.join([x for x in token if x.isalnum() and x.isascii()])
	if len(_token) > 0:
		return _token

	return 'unk'