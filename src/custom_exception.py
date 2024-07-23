"""
Author: Pratyush Khare
"""

class HTSCustomException(Exception):
    """
    Custom exception class for handling specific errors related to the Food Merchant LTR project.
    """

    def __init__(self, message: str):
        """
        Initialize the LTRCustomException with an error message.

        Parameters:
        message (str): The error message to be displayed.
        """
        # Ensure the message is always a string
        if not isinstance(message, str):
            message = str(message)
        super().__init__(message)
