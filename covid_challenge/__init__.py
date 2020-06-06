import os


def get_file_from_s3(s3_client, complete_file_path, cache):
    without_prefix = complete_file_path[5:]

    broken_down = without_prefix.split('/')

    bucket = broken_down[0]

    object = '/'.join(broken_down[1:])

    filename = os.path.join(cache, '-'.join(broken_down[1:]))

    if not os.path.exists(filename):
        for i in range(100):
            try:
                s3_client.download_file(bucket, object, filename)
                break
            except:
                print('There were problems retrieving the file. Attempt {}'.format(i))

    return filename


def put_file_on_s3(s3_client, local_file_path, remote_file_path):
    without_prefix = remote_file_path[5:]

    broken_down = without_prefix.split('/')

    bucket = broken_down[0]

    object = '/'.join(broken_down[1:])

    for i in range(100):
        try:
            s3_client.upload_file(local_file_path, bucket, object)
        except:
            print('there was a problem uploading results to s3. Attempt {}'.format(i))