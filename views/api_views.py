from fastapi.responses import JSONResponse

class APIView:
    @staticmethod
    def success_response(data, message="Success"):
        return JSONResponse(content={"message": message, "data": data}, status_code=200)

    @staticmethod
    def error_response(error_message, status_code=500):
        return JSONResponse(content={"error": error_message}, status_code=status_code)