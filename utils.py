def preprocess_token(token):
	if len(token) == 1:
		return token
		
	_token = ''.join([x for x in token if x.isalnum() and x.isascii()])
	if len(_token) > 0:
		return _token

	return 'UNK'