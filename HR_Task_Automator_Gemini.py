from pdfminer.high_level import extract_text
import google.generativeai as genai
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import plotly.express as px
from openai import OpenAI
import streamlit as st
import pandas as pd
import pytz
import PyPDF2
import ollama
import datetime
import csv
import time
import json
import io
import re
import os


#Initialize some of the state variables at the start of the script:
if "total_resume_processed" not in st.session_state:
    st.session_state["total_resume_processed"] = 1 #Starting number

if "total_resumes" not in st.session_state:
    st.session_state["total_resumes"] = 0 #Get the total resumes for progressbar updates

if 'successful_processed_files' not in st.session_state:
    st.session_state['successful_processed_files'] = []

if 'fail_processed_files' not in st.session_state:
    st.session_state['fail_processed_files'] = []

if 'request_count' not in st.session_state:
    st.session_state['request_count'] = 0

if 'resume_progress_bar' not in st.session_state:
    st.session_state['resume_progress_bar'] = None

#For data filtering and report: START
def get_longest_time_at_a_job_duration_company(json_data):
    """Returns a tuple (company_name, job_role, duration) of the longest time an applicant 
    has worked based on the data of "Company History", handling potential null durations."""
    max_duration = 0
    company_name = ""
    job_role = ""
    for job in json_data["Company History"]:
        duration = job.get("Working Months")  # Use .get() to handle missing keys
        if duration is not None and duration > max_duration:
            max_duration = duration
            company_name = job["Company Name"]
            job_role = job["Job Role"]
    return company_name, job_role, max_duration if max_duration > 0 else None  # Return None if no valid duration found

def validate_folder_path(folder_path):
    # Check if the folder path exists
    if not os.path.exists(folder_path):
        st.toast("Error accessing the directory/path. Please check the folder path.", icon="⚠️")
        return False

    # Check if the folder contains only pdf, doc, or docx files
    valid_extensions = {'.pdf', '.doc', '.docx'}
    all_files_valid = True
    for file in os.listdir(folder_path):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_extensions:
            all_files_valid = False
            st.toast("The folder path contains files other than PDF, DOC, or DOCX.", icon="⚠️")
            break

    return all_files_valid

def get_shortest_time_at_a_job_duration_company(json_data):
    """Returns a tuple (company_name, job_role, duration) of the shortest time an applicant 
    has worked based on the data of "Company History", handling potential null durations."""
    min_duration = float('inf')
    company_name = ""
    job_role = ""
    for job in json_data["Company History"]:
        duration = job.get("Working Months")
        if duration is not None and duration < min_duration:
            min_duration = duration
            company_name = job["Company Name"]
            job_role = job["Job Role"]
    return company_name, job_role, min_duration if min_duration != float('inf') else None 

def get_average_time_at_jobs(json_data):
    """Gets the overall average work duration per job by the applicant 
    based on company history, handling potential null durations."""
    total_months = 0
    valid_job_count = 0
    for job in json_data["Company History"]:
        duration = job.get("Working Months")
        if duration is not None:
            total_months += duration
            valid_job_count += 1
    return total_months / valid_job_count if valid_job_count > 0 else 0

def get_total_number_of_jobs(json_data):
    """Gets the total number of jobs an applicant had based on company history. 
    This function assumes all entries in "Company History" represent valid jobs."""
    return len(json_data["Company History"])

def has_worked_more_than_3_years(json_data):
    """
    Checks if the applicant has worked on any job for more than 3 years (36 months),
    handling potential null durations.
    """
    jobs_more_than_3_years = [job for job in json_data["Company History"] if job.get("Working Months") is not None and job.get("Working Months") > 36]
    has_worked_more_than_3_years = len(jobs_more_than_3_years) > 0
    if has_worked_more_than_3_years:
        average_job_stay = sum(job.get("Working Months") for job in jobs_more_than_3_years) / len(jobs_more_than_3_years)
        num_jobs = len(jobs_more_than_3_years)
        return average_job_stay, num_jobs, has_worked_more_than_3_years
    else:
        return None, None, False

def has_worked_less_than_3_years(json_data):
    """
    Checks if the applicant has worked on any job for less than 3 years (36 months),
    handling potential null durations.
    """
    jobs_less_than_3_years = [job for job in json_data["Company History"] if job.get("Working Months") is not None and job.get("Working Months") < 36]
    has_worked_less_than_3_years = len(jobs_less_than_3_years) > 0
    if has_worked_less_than_3_years:
        average_job_stay = sum(job.get("Working Months") for job in jobs_less_than_3_years) / len(jobs_less_than_3_years)
        num_jobs = len(jobs_less_than_3_years)
        return average_job_stay, num_jobs, has_worked_less_than_3_years
    else:
        return None, None, False
    
def analyze_resume(json_data):
    """Analyzes the resume data and populates an analysis dictionary.

    Args:
        json_data: The JSON data containing the resume information.

    Returns:
        A dictionary containing the analysis results.
    """

    analysis_dict = {
        "Name": json_data["Name"],
        "Total Number of Jobs": get_total_number_of_jobs(json_data),
        "Longest Time at a Job (Duration)": None,  # Placeholder
        "Longest Time at a Job (Company)": None,  # Placeholder
        "Longest Time at a Job (Job Role)": None, #Placeholder
        "Shortest Time at a Job (Duration)": None,  # Placeholder
        "Shortest Time at a Job (Company)": None,  # Placeholder
        "Shortest Time at a Job (Job Role)": None, #Placeholder
        "Average Time at Jobs": get_average_time_at_jobs(json_data),
        "Has worked for more than 3 years": None,  # Placeholder, will be updated
        "Total Number of Jobs worked for more than 3 years": None,  # Placeholder
        "Average Duration of Jobs worked for more than 3 years": None,  # Placeholder
        "Has worked for less than 3 years": None,  # Placeholder
        "Total Number of Jobs worked for less than 3 years": None,  # Placeholder
        "Average Duration of Jobs worked for less than 3 years": None  # Placeholder
    }

    # Get longest job details
    longest_job_company, longest_job_role, longest_job_duration = get_longest_time_at_a_job_duration_company(json_data)
    analysis_dict["Longest Time at a Job (Duration)"] = longest_job_duration
    analysis_dict["Longest Time at a Job (Company)"] = longest_job_company
    analysis_dict["Longest Time at a Job (Job Role)"] = longest_job_role

    # Get shortest job details
    shortest_job_company, shortest_job_role, shortest_job_duration = get_shortest_time_at_a_job_duration_company(json_data)
    analysis_dict["Shortest Time at a Job (Duration)"] = shortest_job_duration
    analysis_dict["Shortest Time at a Job (Company)"] = shortest_job_company
    analysis_dict["Shortest Time at a Job (Job Role)"] = shortest_job_role

    # Get more than 3 years experience details
    avg_duration_gt_3yrs, num_jobs_gt_3yrs, has_worked_gt_3yrs = has_worked_more_than_3_years(json_data)
    analysis_dict["Has worked for more than 3 years"] = has_worked_gt_3yrs
    analysis_dict["Total Number of Jobs worked for more than 3 years"] = num_jobs_gt_3yrs
    analysis_dict["Average Duration of Jobs worked for more than 3 years"] = avg_duration_gt_3yrs

    # Get less than 3 years experience details
    avg_duration_lt_3yrs, num_jobs_lt_3yrs, has_worked_lt_3yrs = has_worked_less_than_3_years(json_data)
    analysis_dict["Has worked for less than 3 years"] = has_worked_lt_3yrs
    analysis_dict["Total Number of Jobs worked for less than 3 years"] = num_jobs_lt_3yrs
    analysis_dict["Average Duration of Jobs worked for less than 3 years"] = avg_duration_lt_3yrs

    return analysis_dict

def is_job_hopper(analysis_dict, job_hopper_average_month_threshold,
                   job_hopper_average_month_less_than_3year_threshold,
                   job_hopper_total_jobs_less_than_3year_threshold, 
                   criteria_method):
    """
    Determines if an applicant is a job hopper based on the provided criteria.

    Args:
        analysis_dict: The dictionary containing the analyzed resume data.
        job_hopper_average_month_threshold: Threshold for average job duration (all jobs).
        job_hopper_average_month_less_than_3year_threshold: Threshold for average job duration (jobs less than 3 years).
        job_hopper_total_jobs_less_than_3year_threshold: Threshold for total number of jobs (less than 3 years).
        criteria_method: How to apply the criteria - "All must be true" or "At least one must be true".

    Returns:
        1 if the applicant is flagged as a job hopper, 0 otherwise.
    """

    condition1 = analysis_dict["Average Time at Jobs"] <= job_hopper_average_month_threshold
    condition2 = analysis_dict["Average Duration of Jobs worked for less than 3 years"] <= job_hopper_average_month_less_than_3year_threshold if analysis_dict["Average Duration of Jobs worked for less than 3 years"] is not None else False
    condition3 = analysis_dict["Total Number of Jobs worked for less than 3 years"] >= job_hopper_total_jobs_less_than_3year_threshold if analysis_dict["Total Number of Jobs worked for less than 3 years"] is not None else False

    if criteria_method == "All criteria must be met":
        is_job_hopper = all([condition1, condition2, condition3])
    elif criteria_method == "At least one criterion must be met":
        is_job_hopper = any([condition1, condition2, condition3])
    elif criteria_method == "Only consider 'Average Time at Jobs'":
        is_job_hopper = condition1
    elif criteria_method == "Only consider 'Avg. Time at Jobs UNDER 3 Years'":
        is_job_hopper = condition2
    elif criteria_method == "Only consider 'Total Jobs UNDER 3 Years'":
        is_job_hopper = condition3
    else:
        raise ValueError("Invalid criteria method.")

    return is_job_hopper

#For data filtering and report: END

#For processing documents from folder path: START

def all_files_in_folder_path_are_valid(folder_path):

    # Check if the folder contains only pdf, doc, or docx files
    valid_extensions = {'.pdf', '.doc', '.docx'}
    all_files_valid = True
    for file in os.listdir(folder_path):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_extensions:
            all_files_valid = False
            break
    
    return all_files_valid

def is_text_based_pdf(file_path):
    """ Check if a PDF file is text-based or scanned/image-based. """
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                if page.extract_text():
                    return True  # Text-based if any text can be extracted
        return False  # Considered image-based if no text was extracted from any page
    except:
        return False  # Return False if there's an error, assuming it can't be processed

def process_documents(folder_path):
    # Scheduler initialization
    continue_handling_documents = True

    while continue_handling_documents:
        start_time = datetime.datetime.now()
        print(f"Start time: {start_time}")
        continue_handling_documents = handle_document_processing(folder_path)
        if not continue_handling_documents: #if done processing
            break
        end_time = datetime.datetime.now()
        print(f"End time: {end_time}")
        duration = end_time - start_time
        duration_in_seconds = duration.total_seconds()
        remaining_time = 80 - duration_in_seconds #remaining time it takes to complete an 80 seconds job #add 20 seconds allowance
        if remaining_time > 0:
            print(f"Sleeping for {remaining_time} seconds to make up the one minute and 20 seconds.")
            time.sleep(remaining_time)
        else:
            print("No need to sleep as the duration or more than 80 seconds.")

    df = pd.DataFrame(st.session_state['job_hopper_all_data'])
    st.session_state['df_job_hopping_results'] = df

def handle_document_processing(folder_path):
    # Check if the maximum requests per minute have been reached
    print("Handling Documents...")

    # File handling and file extension validation here
    valid_extensions = {'.pdf', '.doc', '.docx'}
    total_file_processed_session = 0 #For each session to exit time


    if "total_resumes" not in st.session_state:
        print("Total resumes doesn't exist st.session state currently.")

    #This is for counting the number of documents to process
    if st.session_state["total_resumes"] == 0: #Instantiate it once, it will go back to zero once Run is clicked again
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            file_extension = os.path.splitext(file)[1].lower()

            if file_extension in valid_extensions:
                if file_extension == ".pdf" and not is_text_based_pdf(file_path):
                    continue
                else:
                    st.session_state["total_resumes"] += 1

    
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        file_extension = os.path.splitext(file)[1].lower()

        # Skip files already processed
        if file in st.session_state['successful_processed_files'] or file in st.session_state['fail_processed_files']:
            continue

        if file_extension in valid_extensions:
            total_file_processed_session += 1 ##Incerement the correct processed data to shutdown the scheduler apporpriately
            if file_extension == ".pdf" and not is_text_based_pdf(file_path):
                st.session_state['fail_processed_files'].append(file)
                st.session_state["total_resume_processed"] += 1 #Increment starting number
            else:
                if st.session_state["request_count"] < 14: #Since a minimum of 2 request are needed to complete the job
                    print(f"Running analysis for {file}")  # Debug output
                    response_json_valid = False
                    is_expected_json = False
                    max_attempts = 3
                    parsed_result = {}
                    while not response_json_valid and max_attempts > 0: ## Wait till the response json is valid
                        analysis_result = ""

                        #Test 1
                        try:
                            analysis_result = generate_response(file_path)
                        except Exception as e:
                            print(f"Failed to process the folloiwing file: {file.name}...\n\n Due to error: f{str(e)}")
                            max_attempts = max_attempts - 1 
                            st.toast(f"Failed to process the folloiwing file: {file.name}...\n\n Due to error: f{str(e)}")
                            continue

                        #Test 2
                        parsed_result, response_json_valid = extract_and_parse_json(analysis_result)
                        if response_json_valid == False:
                            print(f"Failed to validate and parse json for {file}... Trying again...")
                            max_attempts = max_attempts - 1
                            continue

                        #Test 3
                        is_expected_json = is_expected_json_content(parsed_result)
                        if is_expected_json == False:
                            print(f"Successfully validated and parse json for {file} but is not expected format... Trying again...")
                            continue

                        #If file successfully passed the tests above
                        st.session_state['successful_processed_files'].append(file)
                        # Debug print the parsed results
                        print(f"Parsed Results for {file}: {parsed_result}")
                        analysis_dict = analyze_resume(parsed_result)
                        analysis_dict["Is Job Hopper?"] = is_job_hopper(
                            analysis_dict,
                            job_hopper_average_month_threshold,
                            job_hopper_average_month_less_than_3year_threshold,
                            job_hopper_total_jobs_less_than_3year_threshold,
                            criteria_method,    
                        )
                        #### Try to save the results on the folder one by one
                        csv_path = os.path.join(folder_path, "job_hopper_results.csv") 
                        try:
                            append_to_csv(analysis_dict, csv_path) #Append the analysis_dict to csv
                        except Exception as e:
                            print(f"Failed to append results of {file} to job_hopper_results.csv")
                            st.toast(f"Warning ⚠️: Failed to append results of {analysis_dict['Name']} to job_hopper_results.csv")
                        processed_documents_status_json_update(os.path.join(folder_path, "processed_documents_status_results.json"))
                        
                        st.session_state['job_hopper_all_data'].append(analysis_dict)
                        st.session_state["applicants_work_history"][analysis_dict["Name"]] = parsed_result
                        st.session_state['resumes_processed_progress_bar'].progress(st.session_state["total_resume_processed"]/st.session_state["total_resumes"], text=f"Analyzed the resume of {st.session_state['total_resume_processed']} applicant/s")
                        print(f'Total_resumes: {st.session_state["total_resumes"]}')
                        print(f'Total_resume_processed: {st.session_state["total_resume_processed"]}')
                        st.session_state["total_resume_processed"] += 1 #Increment starting number
                        applicants_job_history_json_update(os.path.join(folder_path, "applicants_job_history.json"), parsed_result)
                    
                    if max_attempts == 0 and response_json_valid == False and  is_expected_json == False:
                        st.session_state['fail_processed_files'].append(file.name)

    st.session_state['request_count'] = 0 #Refresh to zero, create external code to continue next another minute

    if st.session_state["total_resume_processed"] >= st.session_state["total_resumes"]: #If no other files are needed to be processed this should be assumed to be equal
        print(f'Total_resumes: {st.session_state["total_resumes"]}')
        print(f'Total_resume_processed_with_+1_increment: {st.session_state["total_resume_processed"]}')
        st.toast("Stopping Scheduler. All documents have been processed. Please kindly check.")
        return False #Stop handling documents
    
    return True #continue handling documents

def append_to_csv(data, csv_path):
    """Append a dictionary of data to a CSV file."""
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # Only write header if file does not exist

        writer.writerow(data)

def processed_documents_status_json_update(json_path):
    """Update or create a JSON file with the lists of processed and failed documents."""
    data = {
        "successful_processed_files": st.session_state['successful_processed_files'],
        "fail_processed_files": st.session_state['fail_processed_files']
    }
    try:
        with open(json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(data, jsonfile, indent=4)
    except:
        print("Failed to dump successful and failed processed documents list on processed_documents_status_results.json file")
        st.toast("Failed to update/save successful and failed processed documents list of  processed documents status results")

def applicants_job_history_json_update(json_file_path, new_data):
    # Load existing data from the JSON file
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            data = json.load(file)
    else:
        data = []

    # Append new data to the existing data
    data.append(new_data)

    # Write the updated data back to the JSON file
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)

#For processing documents from folder path: END
@st.experimental_dialog("Resume Json Processing...")
def json_resume_processing():
    st.markdown("## The following example json structure must be strictly followed to successfully process your file.")
    st.write("""\n
[\n
    {\n
        "Name":<name>,\n
        "Year Graduated": <year>,\n
        "Company History" : [\n
            {\n
                "Company Name": <company name>,\n
                "Year Started" <year (number format only)>,\n
                "Job Role": <job role>,\n
                "Working Months": <total months working (number format only)>\n
            }\n
        ]\n
    },\n
    {\n
        "Name":<name>,\n
        "Year Graduated": <year>,\n
        "Company History" : [\n
            {\n
                "Company Name": <company name>,\n
                "Year Started" <year (number format only)>,\n
                "Job Role": <job role>,\n
                "Working Months": <total months working (number format only)>\n
            }\n
        ]\n
    },\n
]\n  
""")
def convert_numeric_values(data):
    if isinstance(data, dict):
        return {k: convert_numeric_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numeric_values(item) for item in data]
    else:
        return convert_to_numeric(data)

def convert_to_numeric(value):
    # Attempt to convert strings that represent integers or floats to their numeric forms
    try:
        return int(value)
    except (ValueError, TypeError):
        pass
    try:
        return float(value)
    except (ValueError, TypeError):
        pass
    return value

def extract_and_parse_json(text):
    # Find the first opening and the last closing curly brackets
    start_index = text.find('{')
    end_index = text.rfind('}')
    
    if start_index == -1 or end_index == -1 or end_index < start_index:
        return None, False  # Proper JSON structure not found

    # Extract the substring that contains the JSON
    json_str = text[start_index:end_index + 1]

    try:
        # Attempt to parse the JSON
        parsed_json = json.loads(json_str)
        # Convert all numeric values if they are in string format
        parsed_json = convert_numeric_values(parsed_json)
        return parsed_json, True
    except json.JSONDecodeError:
        return None, False  # JSON parsing failed

def is_expected_json_content(json_data, type = "job_hopper"):
    """
    Validates if the passed argument is a valid JSON with the expected structure.

    Args:
        json_data: The JSON data to validate.

    Returns:
        True if the JSON is valid and has the expected structure, False otherwise.
    """
    
    try:
        # Try to load the JSON data
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
    except json.JSONDecodeError:
        return False

    if type == "job_hopper":
        # Define required top-level keys for 'job_hopper'
        required_keys = ["Name", "Year Graduated", "Company History"]
        required_company_keys = ["Company Name", "Year Started", "Job Role", "Working Months"]

        if not all(key in data for key in required_keys):
            return False
        if not isinstance(data["Company History"], list):
            return False
        for company_history in data["Company History"]:
            if not all(key in company_history for key in required_company_keys):
                return False
    
    elif type == "job_fit":
        # Define required top-level keys for 'job_fit'
        required_keys = ["Name", "Technical Skills", "Relevant Experience", "Education Relevance", "Overall Score", "Overall Assessment"]
        if not all(key in data for key in required_keys):
            return False
        if not isinstance(data.get("Technical Skills"), int) or not isinstance(data.get("Relevant Experience"), int) or not isinstance(data.get("Education Relevance"), int) or not isinstance(data.get("Overall Score"), int):
            return False
        if not isinstance(data.get("Overall Assessment"), str):
            return False

    else:
        return False  # Unsupported type

    return True  # All checks passed for the specified type

def prepare_data_for_chart(df, names):
    df_filtered = df[df['Name'].isin(names)]
    df_long = df_filtered.melt(id_vars='Name', value_vars=[col for col in df.columns if col not in ['Name', 'Overall Assessment']],
                               var_name='variable', value_name='value')
    return df_long

# Function to generate a spider chart
def generate_spider_chart(data, title):
    # Generate the spider chart using Plotly Express
    fig = px.line_polar(data, r='value', theta='variable', line_close=True,
                        color='Name',  # Differentiating data by candidate
                        title=title, range_r=[0,100])
    fig.update_traces(fill='toself')
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True  # Ensure legend is shown
    )
    return fig 

def generate_response(uploaded_file):
    primary_data_extract_query = f"""
Given the following resume data. I'd like you to extract only the important data points:
1) Name
2) Year Graduated (include the university or college name if available)
4) Job Histories (include company name and the duration)
    - Extract the starting dates and end dates
    - Then manually compute the exact durations. Be accurate with your computations, consider the month and the year.
    - Extract Job Role
    - Do not include or consider internships as part of job history

Notes:
- No yapping!
- If the resume text contains "Present" date, consider that today's date is {datetime.datetime.now().strftime("%B %d, %Y")}.

----------------------------------------
Resume Text:

    """

    query_text = f"""
Given the resume text below, I want you to extract me the following information/data from the resume in a json format:

{{
"Name":<name>,
"Year Graduated": <year>,
"Company History" : [
    {{
        "Company Name": <company name>,
        "Year Started" <year (number format only)>,
        "Job Role": <job role>,
        "Working Months": <total months working (number format only)>
    }}
    ]
}}

Note:
- Please strictly follow the json structure above.
- If the information doesn't exist enter a json "null" value.
- If no company history then pass an empty list.
----------------------------------------

Resume Information:

    """
    # Extract text from the PDF
    if uploaded_file is not None:

    #     ## OLLAMA LLAMA SETUP ###
    #     main_model = "llama3"
    #     text = extract_text(uploaded_file)
    #     primary_prompt = primary_data_extract_query + text

    #     primary_response = ollama.chat(
    #         model=main_model,
    #             messages=[{"role": "system", "content": "You are an HR manager which would like to get the following information from your applicants."}, 
    #                 {"role": "user", "content": primary_prompt }]
    #     )

        
    #     print(f"""Primary Response Text:\n{primary_response["message"]["content"]}""")

    #     main_prompt = query_text + primary_response["message"]["content"]

    #     main_response = ollama.chat(
    #         model=main_model,
    #             messages=[{"role": "system", "content": "You are an HR manager which would like to get the following information from your applicants."}, 
    #                 {"role": "user", "content": primary_prompt },
    #                 {"role":"assistant", "content":primary_response["message"]["content"]},
    #                 {"role": "user", "content": main_prompt}]
    #     )

    #     print(f"""Main Response Text:\n{main_response["message"]["content"]}""")
    
    # return main_response["message"]["content"]

    ### OPENAI GEMINI SETUP ###
    #     main_model = "gpt-3.5-turbo"
    #     client = OpenAI(
    #             api_key = api_key
    #     )
    #     text = extract_text(uploaded_file)
    #     prompt = query_text + text
    #     primary_prompt = primary_data_extract_query + text

    #     primary_response = client.chat.completions.create(
    #         model=main_model,
    #             messages=[{"role": "system", "content": "You are an HR manager which would like to get the following information from your applicants."}, 
    #                 {"role": "user", "content": primary_prompt }],
    #             temperature=0.4
    #     )

        
    #     print(f"Primary Response Text:\n{primary_response.choices[0].message.content}")

    #     main_prompt = query_text + primary_response.choices[0].message.content

    #     main_response = client.chat.completions.create(
    #         model=main_model,
    #             messages=[{"role": "system", "content": "You are an HR manager which would like to get the following information from your applicants."}, 
    #                 {"role": "user", "content": primary_prompt },
    #                 {"role":"assistant", "content":primary_response.choices[0].message.content},
    #                 {"role": "user", "content": main_prompt}],
    #             temperature=0.4
    #     )

    #     print(f"Main Response Text:\n{main_response.choices[0].message.content}")

    # return main_response.choices[0].message.content

        ### GOOGLE GEMINI SETUP ###
        genai.configure(api_key=st.session_state["api_keys"]["GOOGLE_GEN_AI_API_KEY"])
        # Choose a model that's appropriate for your use case.
        model = genai.GenerativeModel('gemini-1.0-pro',
            generation_config=genai.GenerationConfig(
            max_output_tokens=5000,
            temperature=0.4
        ))

        text = extract_text(uploaded_file)
        primary_prompt = primary_data_extract_query + text
        chat = model.start_chat(history=[])
        primary_response = None
        main_response = None

        if st.session_state['request_count'] <= 14:
            primary_response = chat.send_message(primary_prompt)
            # print(f"Primary Response Text:\n{primary_response.text}")
            st.session_state['request_count'] += 1
        else:
            # Raise an exception when the request limit for primary requests is reached
            raise Exception("Request limit reached for initial processing. Please wait a minute before trying again.")

        if st.session_state['request_count'] <= 15:
            main_prompt = query_text + primary_response.text 
            main_response = chat.send_message(main_prompt)
            # print(f"Main Response Text:\n{main_response.text}")
            st.session_state['request_count'] += 1
        else:
            # Raise an exception when the request limit for main requests is reached
            raise Exception("Request limit reached for detailed processing. Please wait a minute before trying again.")
        
    return main_response.text


def parse_response(response):
    result = {}
    lines = response.split('\n')
    for line in lines:
        if ': ' in line:
            key, value = line.split(': ', 1)
            result[key.strip()] = value.strip() if value.strip() else None
    return result

def get_first_numerical_value(text):
    # This function extracts the first numerical value from a given text string.
    if text is None:
        return None
    
    # Check if the input is a numerical value
    if isinstance(text, (int, float)):
        return float(text)
    
    # Ensure the input is a string for regex processing
    text = str(text)
    
    # This regex will find integers or decimals
    numbers = re.findall(r'\d+\.*\d*', text)
    
    return float(numbers[0]) if numbers else None
#FUNCTIONS: END HERE

# Sidebar for API Key and navigation
st.sidebar.title("Settings")
genai_api_key = st.sidebar.text_input("Enter your Google AI Studio API Key", type="password")

if "api_keys" not in st.session_state:
    st.session_state["api_keys"] = {}
st.session_state["api_keys"]["GOOGLE_GEN_AI_API_KEY"] = genai_api_key
selected_option = st.sidebar.radio("Select Task", [
    "Resume Job Hopper Identifier", 
    "Resume - Job Description Fit Identifier", 
    "Resume Data Miner"
])

st.title(selected_option)  # Use the selected option as the page title

# Initialize session state for data storage and file uploads
if 'data' not in st.session_state:
    st.session_state['data'] = []

if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = []

# Conditional content based on selected feature
if selected_option == "Resume Job Hopper Identifier":

    st.markdown("### Job Hopper Criteria")
    job_hopper_average_month_threshold = st.number_input("Average Time at Jobs (Months): Flag if LESS than", min_value=1, value=5)
    job_hopper_average_month_less_than_3year_threshold = st.number_input("Average Time at Jobs UNDER 3 Years (Months): Flag if LESS than", min_value=1, value=5)
    job_hopper_total_jobs_less_than_3year_threshold = st.number_input("Total Jobs UNDER 3 Years: Flag if MORE than", min_value=1, value=5)
    criteria_method = st.radio(
            "How to Flag a Job Hopper:",
            [
                "All criteria must be met", 
                "At least one criterion must be met",
                "Only consider 'Average Time at Jobs'",
                "Only consider 'Avg. Time at Jobs UNDER 3 Years'",
                "Only consider 'Total Jobs UNDER 3 Years'"
            ]
    )
    with st.expander("JSON File Format Uploading Guidelines"):
        st.markdown("#### For uploading JSON files, the following example json structure must be strictly followed to successfully process your file.")
        st.write("""\n
[\n
    {\n
        "Name":<name>,\n
        "Year Graduated": <year>,\n
        "Company History" : [\n
            {\n
                "Company Name": <company name>,\n
                "Year Started" <year (number format only)>,\n
                "Job Role": <job role>,\n
                "Working Months": <total months working (number format only)>\n
            }\n
        ]\n
    },\n
    {\n
        "Name":<name>,\n
        "Year Graduated": <year>,\n
        "Company History" : [\n
            {\n
                "Company Name": <company name>,\n
                "Year Started" <year (number format only)>,\n
                "Job Role": <job role>,\n
                "Working Months": <total months working (number format only)>\n
            }\n
        ]\n
    },\n
]\n  
""")

    folder_path = st.text_input("Enter the folder path:", "")
    uploaded_files = st.file_uploader("Upload Resume PDFs or parse JSON file", accept_multiple_files=True, type=["pdf", "doc", "docx", "json"], key="file_uploader")

    if "df_job_hopping_results" not in st.session_state:
        st.session_state["df_job_hopping_results"] = pd.DataFrame()

    if "applicants_work_history" not in st.session_state:
        st.session_state["applicants_work_history"] = {}

    if uploaded_files is not None:
        st.session_state.uploaded_files = uploaded_files

    if 'job_hopper_all_data' not in st.session_state:
        st.session_state['job_hopper_all_data'] = []

    if st.button('Run'):
        if not st.session_state["api_keys"]["GOOGLE_GEN_AI_API_KEY"] or st.session_state["api_keys"]["GOOGLE_GEN_AI_API_KEY"] == "":
            st.warning("Please enter your Google Gen API Key to proceed.")
            st.stop()

        #Reset values
        st.session_state["applicants_work_history"] = {} #empty session_state
        st.session_state['resumes_processed_progress_bar'] = st.progress(0, text="Analyzing your resume data")
        st.session_state["total_resume_processed"] = 1#Starting number
        st.session_state["total_resumes"] = 0 #Get the total resumes for progressbar updates
        st.session_state['job_hopper_all_data'] = [] #intialize
        st.session_state['successful_processed_files'] = [] #intialize
        st.session_state['fail_processed_files'] = [] #intialize

        if os.path.exists(folder_path):
            print("Uploaded Folder path exists.")
            #Reset the values
            
            if not all_files_in_folder_path_are_valid(folder_path):
                st.toast("The folder path contains files other than PDF, DOC, or DOCX. Proceeding to process the documents with valid exstension files.", icon="⚠️")

            # Process the documents:
            print("Processing Documents")
            process_documents(folder_path)

        else:
            if not st.session_state.uploaded_files:
                st.toast("Error accessing the directory/path. Please check the folder path.", icon="⚠️")


        ###  For processing uploaded pdf files.
        if not st.session_state.uploaded_files and (not folder_path or folder_path == ""):
            st.warning("Please upload at least one PDF file to analyze.")
            
        elif st.session_state.uploaded_files:
            print("Processing uploaded files")
            error_processing_json_flag = True #To show up the dialog once for json files that are not structured properly.

            for file in st.session_state.uploaded_files:
                if file.name.endswith('.json'):
                    resumes_data = json.load(file)
                    st.session_state["total_resumes"] += len(resumes_data)
                else:
                    st.session_state["total_resumes"] += 1 #increment 1 for each pdf or doc files

            continue_handling_documents = True
            while continue_handling_documents:
                start_time = datetime.datetime.now()
                print(f"Start time: {start_time}")
                for file in st.session_state.uploaded_files:
                    if file.name.endswith('.json'):
                        if file.name in st.session_state['successful_processed_files'] or file in st.session_state['fail_processed_files']:
                            continue #skip if already processed last time
                        # The file is a JSON file
                        print("Next Json Processing...")
                        file.seek(0) # since file was opened previously
                        resumes_data = json.load(file)
                        st.session_state['successful_processed_files'].append(file.name) #Automatically consider json files as successfully processed since it doesn't need or require llm content generation
                        for resume in resumes_data:
                            is_expected_json = is_expected_json_content(resume)
                            if not is_expected_json:
                                try:
                                    print(f"Warning ⚠️: The resume of {resume['Name']} doesn't follow the expected format. Please fix it manually")
                                    st.toast(f"Warning ⚠️: The resume of {resume['Name']} doesn't follow the expected format. Please fix it manually")
                                    
                                    if error_processing_json_flag:
                                        json_resume_processing()
                                        error_processing_json_flag = False

                                except Exception as e:
                                    print(f"Warning ⚠️: Error processing resume of json file name: {file.name}")
                                    st.toast(f"Warning ⚠️: Error encountered while processing the json file name: {file.name}")

                            else:
                                analysis_dict = analyze_resume(resume)
                                analysis_dict["Is Job Hopper?"] = is_job_hopper(
                                    analysis_dict,
                                    job_hopper_average_month_threshold,
                                    job_hopper_average_month_less_than_3year_threshold,
                                    job_hopper_total_jobs_less_than_3year_threshold,
                                    criteria_method,
                                )
                                st.session_state['job_hopper_all_data'].append(analysis_dict)
                                st.session_state["applicants_work_history"][analysis_dict["Name"]] = resume
                            st.session_state['resumes_processed_progress_bar'].progress(st.session_state["total_resume_processed"]/st.session_state["total_resumes"], text=f"Analyzed the resume of {st.session_state['total_resume_processed']} applicants")
                            st.session_state["total_resume_processed"] += 1 #Increment starting number

                    else:
                        # Skip files already processed
                        if file.name in st.session_state['successful_processed_files'] or file.name in st.session_state['fail_processed_files']:
                            print("Doc already processed")
                            continue

                        
                        if extract_text(file).strip() == "": #If empty string
                            print(f"Pdf file {file.name} is not text based.")
                            st.session_state['fail_processed_files'].append(file.name)
                            st.session_state["total_resume_processed"] += 1 #Increment starting number
                            continue

                        if st.session_state["request_count"] < 14:
                            # The file is not a JSON file. This could be a pdf, doc, or docs
                            print(f"Running analysis for {file.name}")  # Debug output
                            response_json_valid = False
                            is_expected_json = False
                            max_attempts = 3
                            parsed_result = {}
                            while not response_json_valid and max_attempts > 0: ## Wait till the response json is valid
                                analysis_result = ""

                                #Test 1
                                try:
                                    analysis_result = generate_response(file)
                                except Exception as e:
                                    print(f"Failed to process the following file: {file.name}...\n\n Due to error: f{str(e)}.\n\n Trying again... Retries left: {max_attempts} attempt/s")
                                    max_attempts = max_attempts - 1 
                                    st.toast(f"Failed to process the following file: {file.name}...\n\n Due to error: f{str(e)}.\n\n Trying again... Retries left: {max_attempts} attempt/s")
                                    continue

                                #Test 2
                                parsed_result, response_json_valid = extract_and_parse_json(analysis_result)
                                if response_json_valid == False:
                                    print(f"Failed to validate and parse json for {file.name}... Trying again...")
                                    max_attempts = max_attempts - 1
                                    continue

                                #Test 3
                                is_expected_json = is_expected_json_content(parsed_result)
                                if is_expected_json == False:
                                    print(f"Successfully validated and parse json for {file.name} but is not expected format... Trying again...")
                                    continue

                                #If file successfully passed the tests above
                                st.session_state['successful_processed_files'].append(file.name)
                                # Debug print the parsed results
                                print(f"Parsed Results for {file.name}: {parsed_result}")
                                analysis_dict = analyze_resume(parsed_result)
                                analysis_dict["Is Job Hopper?"] = is_job_hopper(
                                        analysis_dict,
                                        job_hopper_average_month_threshold,
                                        job_hopper_average_month_less_than_3year_threshold,
                                        job_hopper_total_jobs_less_than_3year_threshold,
                                        criteria_method,
                                )

                                if os.path.exists(folder_path): #Only if a valid folder path were given
                                    csv_path = os.path.join(folder_path, "job_hopper_results.csv") 
                                    try:
                                        append_to_csv(analysis_dict, csv_path) #Append the analysis_dict to csv
                                    except Exception as e:
                                        print(f"Failed to append results of {file} to job_hopper_results.csv")
                                        st.toast(f"Warning ⚠️: Failed to append results of {analysis_dict['Name']} to job_hopper_results.csv")
                                    processed_documents_status_json_update(os.path.join(folder_path, "processed_documents_status_results.json"))
                        
                                st.session_state['job_hopper_all_data'].append(analysis_dict)
                                st.session_state["applicants_work_history"][analysis_dict["Name"]] = parsed_result
                                st.session_state['resumes_processed_progress_bar'].progress(st.session_state["total_resume_processed"]/st.session_state["total_resumes"], text=f"Analyzed the resume of {st.session_state['total_resume_processed']} applicant/s")
                                st.session_state["total_resume_processed"] += 1 #Increment starting number

                                if os.path.exists(folder_path): #Only if a valid folder path were given
                                    applicants_job_history_json_update(os.path.join(folder_path, "applicants_job_history.json"), parsed_result)
                            
                            if max_attempts == 0 and response_json_valid == False and  is_expected_json == False:
                                st.session_state['fail_processed_files'].append(file.name)

                st.session_state['request_count'] = 0 #Refresh to zero, create external code to continue next another minute
                
                if st.session_state["total_resume_processed"] >= st.session_state["total_resumes"]: #If no other files are needed to be processed this should be assumed to be equal
                    print(f'Total_resumes: {st.session_state["total_resumes"]}')
                    print(f'Total_resume_processed_with_+1_increment: {st.session_state["total_resume_processed"]}')
                    st.toast("Stopping Scheduler. All documents have been processed. Please kindly check.")
                    continue_handling_documents = False
                    break #break while loop
                
                end_time = datetime.datetime.now()
                print(f"End time: {end_time}")
                duration = end_time - start_time
                duration_in_seconds = duration.total_seconds()
                remaining_time = 80 - duration_in_seconds #remaining time it takes to complete an 80 seconds job #add 20 seconds allowance
                if remaining_time > 0:
                    print(f"Sleeping for {remaining_time} seconds to make up the one minute and 20 seconds.")
                    time.sleep(remaining_time)
                else:
                    print("No need to sleep as the duration or more than 80 seconds.")

            df = pd.DataFrame(st.session_state['job_hopper_all_data'])
            st.session_state['df_job_hopping_results'] = df
            
    
    if st.session_state['df_job_hopping_results'].empty == False:   
        st.session_state['resumes_processed_progress_bar'].empty()
        st.write("Analysis Results")
        df = st.session_state['df_job_hopping_results']
        st.dataframe(df)

        # Download links for CSV and Excel
        csv = df.to_csv(index=False).encode('utf-8')
        st.write("_Note: The durations are calculated/expressed in terms of months._")
        csv_downloaded = st.download_button("Download as CSV", csv, 'resume_job_hoppers.csv', 'text/csv')
        excel = io.BytesIO()
        with pd.ExcelWriter(excel, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_downloaded = st.download_button("Download data as Excel", excel.getvalue(), 'resume_job_hoppers..xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


        # Preparing JSON data for successfully and fail processed files
        processed_files = {
            "successful_processed_files": st.session_state['successful_processed_files'],
            "fail_processed_files": st.session_state['fail_processed_files']
        }
        json_str = json.dumps(processed_files, indent=4).encode('utf-8')
        
        # Download link for JSON
        json_downloaded = st.download_button("Download Processed Files Status JSON", json_str, 'processed_files.json', mime="application/json")

        # Convert the dictionary to a list of dictionaries including the name inside each dictionary
        applicant_list = [{**info} for name, info in st.session_state["applicants_work_history"].items()]
        
        # Convert the list into JSON
        json_str = json.dumps(applicant_list, indent=4)
        
        # Create a Streamlit download button for the JSON data
        st.download_button(
            label="Download Applicants Job History JSON",
            data=json_str,
            file_name="applicants_work_history.json",
            mime="application/json"
        )
        
        applicant_names = list(st.session_state["applicants_work_history"].keys())
        selected_applicant = st.selectbox("Select Applicant", applicant_names)

        # Display selected applicant's details in an expander
        if selected_applicant:
            with st.expander(f"Details for {selected_applicant}"):
                applicant_details = st.session_state["applicants_work_history"][selected_applicant]
                st.json(applicant_details)

        # Only show balloons if either download button is clicked
        if csv_downloaded or excel_downloaded:
            st.balloons()


elif selected_option == "Resume - Job Description Fit Identifier":
    
    # Input for job description
    job_description = st.text_area("Enter the Job Description", height=300)

    if "df_job_fit_results" not in st.session_state:
        st.session_state["df_job_fit_results"] = pd.DataFrame()

    # Dynamic criteria input with removal option
    if 'criteria' not in st.session_state:
        st.session_state['criteria'] = {
            "tech_skills_weight": 50,  # Default value for the slider
            "experience_weight": 50,   # Default value for the slider
            "education_weight": 50     # Default value for the slider
        }
    
    # Form start
    with st.container(border=True):
        st.markdown("### Criteria Configuration")

        # Technical Skills
        st.markdown("##### Technical Skills")
        st.markdown("_Description:_ Technical skills are the specific competencies required to perform tasks efficiently in a job role. This may include software proficiency, technical methodologies, operational skills, or any specialized abilities pertinent to the position, regardless of industry.")
        st.session_state['criteria']["tech_skills_weight"] = st.slider("Weight for Technical Skills (1-100)%", 1, 100, key='tech_skills_weight')

        # Dynamic keyword management for Technical Skills
        with st.expander("Key Criteria for Evaluation - Technical Skills"):
            keyword_input = st.text_input("Add a new keyword", key='tech_skill_keyword')
            if st.button("Add Keyword", key='add_tech_keyword'):
                if 'tech_keywords' not in st.session_state:
                    st.session_state.tech_keywords = []
                st.session_state.tech_keywords.append(keyword_input)
            if 'tech_keywords' in st.session_state:
                for idx, keyword in enumerate(st.session_state.tech_keywords):
                    st.markdown(f"- {keyword}")
                    if st.button(f"Remove '{keyword}'", key=f'remove_tech_{idx}'):
                        st.session_state.tech_keywords.remove(keyword)

        # Relevant Experience
        st.markdown("##### Relevant Experience")
        st.markdown("_Description:_ Relevant experience refers to the practical application of skills in previous employment that directly aligns with the responsibilities and functions of the job role being applied for.")
        st.session_state['criteria']["experience_weight"] = st.slider("Weight for Relevant Experience (1-100)%", 1, 100, key='experience_weight')

        # Dynamic keyword management for Relevant Experience
        with st.expander("Key Criteria for Evaluation - Relevant Experience"):
            keyword_input = st.text_input("Add a new keyword", key='exp_keyword')
            if st.button("Add Keyword", key='add_exp_keyword'):
                if 'exp_keywords' not in st.session_state:
                    st.session_state.exp_keywords = []
                st.session_state.exp_keywords.append(keyword_input)
            if 'exp_keywords' in st.session_state:
                for idx, keyword in enumerate(st.session_state.exp_keywords):
                    st.markdown(f"- {keyword}")
                    if st.button(f"Remove '{keyword}'", key=f'remove_exp_{idx}'):
                        st.session_state.exp_keywords.remove(keyword)

        # Education Relevance
        st.markdown("##### Education Relevance")
        st.markdown("_Description:_ Education relevance encompasses the formal educational qualifications and training that are necessary or beneficial for the job role.")
        st.session_state['criteria']["education_weight"] = st.slider("Weight for Education Relevance (1-100)%", 1, 100, key='education_weight')

        # Dynamic keyword management for Education Relevance
        with st.expander("Key Criteria for Evaluation - Education Relevance"):
            keyword_input = st.text_input("Add a new keyword", key='edu_keyword')
            if st.button("Add Keyword", key='add_edu_keyword'):
                if 'edu_keywords' not in st.session_state:
                    st.session_state.edu_keywords = []
                st.session_state.edu_keywords.append(keyword_input)
            if 'edu_keywords' in st.session_state:
                for idx, keyword in enumerate(st.session_state.edu_keywords):
                    st.markdown(f"- {keyword}")
                    if st.button(f"Remove '{keyword}'", key=f'remove_edu_{idx}'):
                        st.session_state.edu_keywords.remove(keyword)

        #### Change the code below...

    uploaded_files = st.file_uploader("Upload Resume PDFs", accept_multiple_files=True, type="pdf", key="file_uploader_fit")
    st.session_state.uploaded_files = uploaded_files

    if st.button('Analyze Resumes'):
        if not st.session_state["api_keys"]["GOOGLE_GEN_AI_API_KEY"] or st.session_state["api_keys"]["GOOGLE_GEN_AI_API_KEY"] == "":
            st.warning("Please enter your Google Gen API Key to proceed.")
            st.stop()

        if not uploaded_files:
            st.warning("Please upload at least one PDF file to analyze.")
        elif not st.session_state.get('criteria'):
            st.warning("Please add at least one criterion.")
        else:
            st.session_state['successful_processed_files'] = [] #intialize
            st.session_state['fail_processed_files'] = [] #intialize
            st.session_state['resumes_processed_progress_bar'] = st.progress(0, text="Analyzing your resume data")
            st.session_state["total_resumes"] = len(st.session_state.uploaded_files)
            st.session_state["total_resume_processed"] = 1#Starting number

            continue_handling_documents = True
            while continue_handling_documents:
                start_time = datetime.datetime.now()
                print(f"Start time: {start_time}")
                all_data = []
                for uploaded_file in uploaded_files:
                    text = extract_text(uploaded_file)
                    prompt = f"""Please evaluate the following resume based on the job description and criteria provided. Score the applicant from the range of (0-100) based on the criteria given.
                    The weight will be only be used as future basis for hiring the applicant, but you can use this as a guide for how critical a criteria should be scored by you.\n\n
                    Resume text: {text}\n\nJob Description: {job_description}\n\nHere are the weights of each of the criteria and the main keywords to look for:\n"""
                    
                    tech_keywords_formatted = "\n".join(st.session_state.tech_keywords)
                    exp_keywords_formatted = "\n".join(st.session_state.exp_keywords)
                    edu_keywords_formatted = "\n".join(st.session_state.edu_keywords)

                    # Now, use these variables directly in the f-string without backslashes in the expressions
                    prompt += f"""
                    1) Technical Skills: (Weight: {st.session_state['criteria']["tech_skills_weight"]}, Scoring: (0-100))
                    Main Keywords for Evaluation:
                    {tech_keywords_formatted}

                    2) Relevant Experience: (Weight: {st.session_state['criteria']["experience_weight"]}, Scoring: (0-100))
                    Main Keywords for Evaluation:
                    {exp_keywords_formatted}

                    3) Education Relevance: (Weight: {st.session_state['criteria']["education_weight"]}, Scoring: (0-100))
                    Main Keywords for Evaluation:
                    {edu_keywords_formatted}
                    """

                    prompt += f"""\n
                    ------------------------
                    Provide your answer and scores in a json format following the structure below:
                    {{
                        "Name": <applicant's name>,
                        "Technical Skills": <score (number only)>,
                        "Relevant Experience": <score (number only)>,
                        "Education Relevance": <score (number only)>,
                        "Overall Score":<score (0-100) (number only)>,
                        "Overall Assessment":<a text description or report on the overall assessment of the applicant>
                    }}
                    ------------------------\n
                    """

                    genai.configure(api_key=st.session_state["api_keys"]["GOOGLE_GEN_AI_API_KEY"])
                    # Choose a model that's appropriate for your use case.
                    model = genai.GenerativeModel('gemini-1.5-flash',
                        generation_config=genai.GenerationConfig(
                        max_output_tokens=5000,
                        temperature=0.4,
                        response_mime_type = "application/json"
                    ))

                    if st.session_state["request_count"] < 15:
                        response_json_valid = False
                        is_expected_json = False
                        max_attempts = 3
                        parsed_result = {}

                        while not response_json_valid and max_attempts > 0: ## Wait till the response json is valid
                            response = ""
    
                            st.session_state['request_count'] += 1
                            max_attempts = max_attempts - 1 
                            try:
                                response = model.generate_content(prompt).text
                            except Exception as e:
                                print(f"Failed to process the following file: {uploaded_file.name}...\n\n Due to error: f{str(e)}.\n\n Trying again... Retries left: {max_attempts} attempt/s")
                                st.toast(f"Failed to process the following file: {uploaded_file.name}...\n\n Due to error: f{str(e)}.\n\n Trying again... Retries left: {max_attempts} attempt/s")
                                continue #Continue on next operations, this will skip the while loop entirely

                            parsed_result, response_json_valid = extract_and_parse_json(response)
                            if response_json_valid == False:
                                print(f"Failed to validate and parse json for {uploaded_file.name}... Trying again...")
                                max_attempts = max_attempts - 1
                                continue

                            is_expected_json = is_expected_json_content(parsed_result, type = "job_fit")
                            if is_expected_json == False:
                                print(f"Successfully validated and parse json for {uploaded_file.name} but is not expected format... Trying again...")
                                print(f"Please review results: {parsed_result}\n\n")
                                continue

                            st.session_state['successful_processed_files'].append(uploaded_file.name)
                            print(f"Parsed Results for {uploaded_file.name}: {parsed_result}")
                            all_data.append(parsed_result)
                            st.session_state['resumes_processed_progress_bar'].progress(st.session_state["total_resume_processed"]/st.session_state["total_resumes"], text=f"Analyzed the resume of {st.session_state['total_resume_processed']} applicant/s")
                            st.session_state["total_resume_processed"] += 1 #Increment starting number


                        if max_attempts == 0 and response_json_valid == False and is_expected_json == False:
                            st.session_state['fail_processed_files'].append(uploaded_file.name)

                st.session_state['request_count'] = 0 #Reset request_count

            ##Add timer for request per 1 minute handling
            ### Show the successful and failed processed files

                if st.session_state["total_resume_processed"] >= st.session_state["total_resumes"]: #If no other files are needed to be processed this should be assumed to be equal
                    print(f'Total_resumes: {st.session_state["total_resumes"]}')
                    print(f'Total_resume_processed_with_+1_increment: {st.session_state["total_resume_processed"]}')
                    st.toast("Stopping Scheduler. All documents have been processed. Please kindly check.")
                    continue_handling_documents = False
                    break #break while loop

                end_time = datetime.datetime.now()
                print(f"End time: {end_time}")
                duration = end_time - start_time
                duration_in_seconds = duration.total_seconds()
                remaining_time = 80 - duration_in_seconds #remaining time it takes to complete an 80 seconds job #add 20 seconds allowance
                if remaining_time > 0:
                    print(f"Sleeping for {remaining_time} seconds to make up the one minute and 20 seconds.")
                    time.sleep(remaining_time)
                else:
                    print("No need to sleep as the duration or more than 80 seconds.")
            
            df = pd.DataFrame(all_data)
            ### Dummy data sample
            # data = {
            #     "Name": ["Name 1", "Name 2", "Name 3"],
            #     "Academic Achievement": [92, 88, 95],  # Hypothetical academic scores out of 100
            #     "Technical Skills": [68, 69, 87],  # Skills rating out of 10
            #     "Relevant Experience (years)": [72, 65, 93],  # Number of years of relevant experience
            #     "Leadership": [61, 70, 81]  # Binary indicator of leadership experience
            # }
            # df = pd.DataFrame(data)
            st.session_state['df_job_fit_results'] = df
            
    if st.session_state["df_job_fit_results"].empty == False:
        st.session_state['resumes_processed_progress_bar'].empty()
        st.write("Scoring Results")
        df = st.session_state['df_job_fit_results']
        st.dataframe(df)
        # Download links for CSV
        csv = df.to_csv(index=False).encode('utf-8')
        csv_downloaded = st.download_button("Download data as CSV", csv, 'job_fit_analysis.csv', 'text/csv')
        excel = io.BytesIO()
        with pd.ExcelWriter(excel, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_downloaded = st.download_button("Download data as Excel", excel.getvalue(), 'job_fit_analysis.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        # Only show balloons if either download button is clicked
        if csv_downloaded or excel_downloaded:
            st.balloons()

    if st.session_state["df_job_fit_results"].empty == False:
        df = st.session_state['df_job_fit_results']
        # Button to trigger the chart generation
        st.markdown("### Candidates Evaluation and Comparison")
        selected_names = st.multiselect('Select Candidates', df['Name'])

        if st.button('Generate Chart'):
            if selected_names:
                # Prepare data for the selected candidates
                data_long = prepare_data_for_chart(df, selected_names)
                # Generate the spider chart
                fig = generate_spider_chart(data_long, 'Comparative Spider Chart for Selected Candidates')
                # Display the spider chart
                st.plotly_chart(fig, use_container_width=True)
                st.balloons()
            else:
                st.warning("Please select at least one candidate to analyze.")

    if st.session_state["df_job_fit_results"].empty == False:
        df = st.session_state['df_job_fit_results']
        st.markdown("### Select a candidate for overall assessment.")
        selected_applicant = st.selectbox("Select Applicant", df["Name"])

        if selected_applicant:
            with st.expander("Applicant Feedback"):
                # Fetch the 'Overall Assessment' for the selected applicant
                assessment = df[df['Name'] == selected_applicant]['Overall Assessment'].iloc[0]  # Get the first entry if there are duplicates
                st.write(assessment)

elif selected_option == "Resume Data Miner":

    if "df_resume_mine_results" not in st.session_state:
        st.session_state["df_resume_mine_results"] = pd.DataFrame()

    # Input for defining what data to mine
    if 'data_points' not in st.session_state:
        st.session_state['data_points'] = []

    with st.form("data_point_form"):
        new_data_point = st.text_input("Enter data point to extract from resume (e.g., 'Email', 'Name', 'Phone Number')")
        add_data_point = st.form_submit_button("Add Data Point")

    if add_data_point and new_data_point:
        st.session_state['data_points'].append(new_data_point)

    # Display current data points with option to remove
    if st.session_state['data_points']:
        for idx, point in enumerate(st.session_state['data_points']):
            st.text(f"Data Point {idx + 1}: {point}")
            if st.button(f"Remove", key=f"remove_{point}"):
                st.session_state['data_points'].remove(point)
                st.experimental_rerun()

    uploaded_files = st.file_uploader("Upload Resume PDFs", accept_multiple_files=True, type="pdf", key="file_uploader_miner")

    if st.button('Mine Resumes'):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file to analyze.")
        elif not st.session_state.get('data_points'):
            st.warning("Please add at least one data point to extract.")
        else:
            all_results = []
            for uploaded_file in uploaded_files:
                text = extract_text(uploaded_file)
                prompt = f"""Extract the following information from the resume, enter NA if it doesn't exist:\n\nResume text: {text}\n\n"""
                for point in st.session_state['data_points']:
                    prompt += f"{point}\n"
                prompt += f"""\n
                ------------------------
                Provide your answer in a json format following the structure below:
                {{
                """ 
                for point in st.session_state['data_points']:
                    prompt += f"\"{point}\": <information>"
                    if idx < len(st.session_state['criteria']):
                        prompt += ","
                prompt += "\n"
                prompt += f"""}}
                --------------
                """

                genai.configure(api_key=st.session_state["api_keys"]["GOOGLE_GEN_AI_API_KEY"])
                # Choose a model that's appropriate for your use case.
                model = genai.GenerativeModel('gemini-1.5-flash',
                    generation_config=genai.GenerationConfig(
                    max_output_tokens=3000,
                    temperature=0.4,
                    response_mime_type="application/json"
                ))
            
        
                response_json_valid = False
                max_attempts = 3
                parsed_result = {}

                while not response_json_valid and max_attempts >0: ## Wait till the response json is valid
                    response = model.generate_content(prompt).text
                    parsed_result, response_json_valid = extract_and_parse_json(response)
                    if response_json_valid == False:
                        print(f"Failed to validate and parse json for {uploaded_file.name}... Trying again...")
                        max_attempts = max_attempts - 1 

                print(f"Parsed Results for {uploaded_file.name}: {parsed_result}")
                all_results.append(parsed_result)


            result_df = pd.DataFrame(all_results)
            st.session_state['df_resume_mine_results'] = result_df
    
    if st.session_state["df_resume_mine_results"].empty == False:
        st.write("Extraction Results")
        df = st.session_state['df_resume_mine_results']
        st.dataframe(df)
        # Download links for CSV
        csv = df.to_csv(index=False).encode('utf-8')
        csv_downloaded = st.download_button("Download as CSV", csv, 'extracted_resume_information.csv', 'text/csv')
        excel = io.BytesIO()
        with pd.ExcelWriter(excel, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_downloaded = st.download_button("Download data as Excel", excel.getvalue(), 'extracted_resume_information.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        # Only show balloons if either download button is clicked
        if csv_downloaded or excel_downloaded:
            st.balloons()

