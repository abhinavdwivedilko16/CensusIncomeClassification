import sys
from src.logger import logging

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occourred in python script [{0}] line number [{1}] error message [{2}]".format(
    file_name,exc_tb.tb_lineno,str(error))

    return error_message
    

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)       # we are inheriting from the exception so this super is used
        self.error_message=error_message_detail(error_message,error_detail=error_detail)       #here we are giving parameter for custom exception
    
    def __str__(self):
        return self.error_message
    
#if __name__=="__main__":
#    try:
#       a=1/0
#    except Exception as e:
#        logging.info("Divide by zero")
#        raise CustomException(e,sys)