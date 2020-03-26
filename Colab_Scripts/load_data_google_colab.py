# Sript for download the files
from google.colab import auth
from googleapiclient.discovery import build
import io , requests, os
import sys
auth.authenticate_user()
from googleapiclient.discovery import build

drive_service = build('drive', 'v3')

def get_parent_folder(folder_name):
  page_token = None
  folder_array = []
  query = "name='%s' and mimeType='application/vnd.google-apps.folder'" % folder_name
  while True:
      response = drive_service.files().list(q=query,
                                          spaces='drive',
                                          fields='nextPageToken, files(id, name)',
                                          pageToken=page_token).execute()
      for file in response.get('files', []):
          # Process change
          #print (file.get('name'), file.get('id'))
          folder_array.append({"name" : file.get('name'), "id" : file.get('id')})
      page_token = response.get('nextPageToken', None)
      if page_token is None:
          break
  return folder_array

def get_files_from_parent(parent_id):
  page_token = None
  folder_array = dict()
  query = "'%s' in parents" % parent_id
  while True:
      response = drive_service.files().list(q=query,
                                          spaces='drive',
                                          fields='nextPageToken, files(id, name)',
                                          pageToken=page_token).execute()
      for file in response.get('files', []):
          # Process change
          folder_array.update({file.get('name'):file.get('id')})
      page_token = response.get('nextPageToken', None)
      if page_token is None:
          break
  return folder_array

def get_file_buffer(file_id, verbose=0):
  from googleapiclient.http import MediaIoBaseDownload
  request = drive_service.files().get_media(fileId=file_id)
  downloaded = io.BytesIO()
  downloader = MediaIoBaseDownload(downloaded, request)
  done = False
  while done is False:
    # _ is a placeholder for a progress object that we ignore.
    # (Our file is small, so we skip reporting progress.)
    progress, done = downloader.next_chunk()
    if verbose:
      sys.stdout.flush()
      sys.stdout.write('\r')
      percentage_done = progress.resumable_progress * 100/progress.total_size
      sys.stdout.write("[%-100s] %d%%" % ('='*int(percentage_done), int(percentage_done)))
  downloaded.seek(0)
  return downloaded

import shutil
def downloads_files_and_data(input_file_meta):
  SOURCE_FOLDER='/content/'
  print('********** Download Logics Files **********')
  for file, id in input_file_meta.items():
    # If file is data -> need to go inside and download
    if file == 'data':
      datas = get_files_from_parent(id).items()
      # Remove and create data folder
      if(os.path.exists('data')):
        shutil.rmtree('data', ignore_errors=True)
      os.makedirs('data')
      for file, id in datas:
        DATA_FOLDER = '/content/data'
        print('********** Download Data Files **********')
        try:
          downloadedData = get_file_buffer(id, verbose=1)
          dest_data = os.path.join(DATA_FOLDER, file)
          print("processing %s data" % file)
          with open(dest_data, "wb") as out:
            out.write(downloaded.read())
            print("Done data %s" % dest_data)
        except ValueError:
          print('SOME_THING_WENT_WRONG', ValueError)
    else:
      try:
        downloaded = get_file_buffer(id, verbose=1)
      except ValueError:
        print('SOME_THING_WENT_WRONG', ValueError)
      dest_file = os.path.join(SOURCE_FOLDER, file)
      print("processing %s data" % file)
      with open(dest_file, "wb") as out:
        out.write(downloaded.read())
        print("Done logic file %s" % dest_file)

from google.colab import files
folder_name = 'Data'
parent_folder = get_parent_folder(folder_name)
if parent_folder and len(parent_folder) >= 1:
  input_file_meta = get_files_from_parent(parent_folder[0]["id"])
else:
  sys.exit()
downloads_files_and_data(input_file_meta)
!ls