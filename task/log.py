import logging


def global_logger(name):
	"""
	Definition for custom logger method.
	Default log level is set to INFO.

	Args:
	name: module name

	Return:
	logger: logger object
	"""
	
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	
	handler = logging.StreamHandler()
	handler.setFormatter(formatter)
	
	logger = logging.getLogger(name)
	logger.setLevel(logging.DEBUG)
	
	if not logger.handlers:
		logger.addHandler(handler)
	return logger