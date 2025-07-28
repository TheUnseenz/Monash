import os
import subprocess

def convert_pptx_to_pdf_libreoffice():
    """
    Converts all .pptx files from the 'PowerpointSlides' folder
    (where this script is located) to .pdf files in a sibling 'PdfSlides' folder.
    Requires LibreOffice installed.
    """
    # Get the directory where the script itself is located
    script_directory = os.path.dirname(os.path.abspath(__file__))
    input_folder = script_directory # .pptx files are in the same folder as the script

    # Determine the parent directory (e.g., 'UnitContents')
    parent_directory = os.path.dirname(script_directory)

    # Define the output folder path (e.g., 'UnitContents/PdfSlides')
    output_folder = os.path.join(parent_directory, "PdfSlides")

    print(f"Input .pptx files from: {input_folder}")
    print(f"Output .pdf files to: {output_folder}")

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # --- Configure LibreOffice Path ---
    # Adjust this path if LibreOffice is not found automatically on your system.
    # Common paths:
    # Windows: r"C:\Program Files\LibreOffice\program\soffice.exe"
    # macOS: "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    # Linux: "/usr/bin/libreoffice" (often in PATH by default)

    libreoffice_path = None
    if os.name == 'nt': # Windows
        default_paths = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"
        ]
        for path in default_paths:
            if os.path.exists(path):
                libreoffice_path = path
                break
    elif os.uname().sysname == 'Darwin': # macOS
        libreoffice_path = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    else: # Linux and other Unix-like systems
        libreoffice_path = "/usr/bin/libreoffice"

    if not libreoffice_path or not os.path.exists(libreoffice_path):
        print("Error: LibreOffice executable not found.")
        print("Please ensure LibreOffice is installed and its 'soffice' executable path")
        print(f"is correctly set in the script (current attempt: {libreoffice_path}).")
        return

    print(f"Using LibreOffice executable: {libreoffice_path}")

    converted_count = 0
    skipped_count = 0
    error_count = 0

    # Iterate through all files in the input directory
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".pptx"):
            input_pptx_path = os.path.join(input_folder, filename)
            output_pdf_filename = os.path.splitext(filename)[0] + ".pdf"
            output_pdf_path = os.path.join(output_folder, output_pdf_filename)

            # Check if the PDF already exists in the output folder
            if os.path.exists(output_pdf_path):
                print(f"  Skipping '{filename}': Corresponding PDF '{output_pdf_filename}' already exists in '{output_folder}'.")
                skipped_count += 1
                continue

            print(f"  Converting '{filename}'...")
            try:
                # Construct the command for LibreOffice conversion
                command = [
                    libreoffice_path,
                    '--headless',           # Run LibreOffice without a GUI
                    '--convert-to', 'pdf',  # Specify output format as PDF
                    '--outdir', output_folder, # Output directory (the new PdfSlides folder)
                    input_pptx_path         # Input .pptx file
                ]

                # Execute the command
                # capture_output=True captures stdout/stderr, text=True decodes as text
                # check=True raises CalledProcessError if the command returns a non-zero exit code
                result = subprocess.run(command, capture_output=True, text=True, check=True)

                converted_count += 1
                print(f"  Successfully converted '{filename}'.")
                if result.stdout:
                    print(f"    LibreOffice Output: {result.stdout.strip()}")
                if result.stderr:
                    print(f"    LibreOffice Warnings/Errors: {result.stderr.strip()}")

            except subprocess.CalledProcessError as e:
                print(f"  Error converting '{filename}': Command failed with exit code {e.returncode}")
                print(f"    STDOUT: {e.stdout.strip()}")
                print(f"    STDERR: {e.stderr.strip()}")
                error_count += 1
            except FileNotFoundError:
                print(f"  Error: LibreOffice executable not found at '{libreoffice_path}'.")
                print("  Please check the 'libreoffice_path' in the script.")
                error_count += 1
                break # Exit the loop if LibreOffice isn't found
            except Exception as e:
                print(f"  An unexpected error occurred for '{filename}': {e}")
                error_count += 1

    print("\n--- Conversion Summary ---")
    print(f"Total .pptx files found: {converted_count + skipped_count + error_count}")
    print(f"Successfully converted: {converted_count}")
    print(f"Skipped (PDF already exists): {skipped_count}")
    print(f"Failed conversions: {error_count}")

if __name__ == "__main__":
    convert_pptx_to_pdf_libreoffice()
