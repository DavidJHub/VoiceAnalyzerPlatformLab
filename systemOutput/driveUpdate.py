from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import pandas as pd

def append_rows_to_google_sheet(file_path, spreadsheet_id, sheet_name="Sheet1"):
    """
    Appends rows from an Excel (.xlsx) file to the end of the data table in a Google Sheets sheet.

    Args:
        file_path (str): Path to the Excel file.
        spreadsheet_id (str): ID of the Google Sheets file.
        sheet_name (str): Name of the sheet in the Google Sheets file (default "Sheet1").

    Returns:
        None
    """
    # Authenticate using the service account
    credentials = Credentials.from_service_account_file('grand-citadel-447320-v4-6686652842a5.json')
    service = build('sheets', 'v4', credentials=credentials)

    # Read the Excel file
    data = pd.read_excel(file_path)

    # Convert dates to strings in ISO 8601 format (YYYY-MM-DD)
    for column in data.select_dtypes(include=['datetime64[ns]', 'datetime']):
        data[column] = data[column].dt.strftime('%Y-%m-%d')

    # Replace NaN, None, and empty values with 0
    data = data.fillna(0)  # Replace missing values with 0

    # Convert the DataFrame to a list of lists (format required by Google Sheets API)
    rows = data.values.tolist()

    # Get the current data in the sheet
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=f"{sheet_name}"
    ).execute()

    existing_values = result.get("values", [])
    last_row = len(existing_values)  # Get the last row with data

    # Prepare the range to append the rows at the end of the table
    start_row = last_row + 1

    # Append data to the sheet
    request = service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"{sheet_name}!A{start_row}",
        valueInputOption="RAW",
        body={"values": rows}
    )

    response = request.execute()
    print(f"Data appended successfully at row {start_row}. Response: {response}")

if __name__ == "__main__":
    file_path = 'VAP_REPORT.xlsx'
    spreadsheet_id = '1RfG5fSR7mSP5soncCORCI4Zjas_0X-IHlW9OqZ-cC80'
    sheet_name = "dailyreport"
    append_rows_to_google_sheet(file_path, spreadsheet_id, sheet_name)
