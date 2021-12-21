# HateFlow backend
HateFlow can detect and classify inappropriate comments. It is available using an API documented [here](http://docs.hateflow.de).
## Installation
(tested on Ubuntu 20.04 and Python 3.8)

```shell
cd hateflow
pip3 install -r requirements.txt  --no-cache-dir
```

## Training a model
```shell
python3 train.py
```

## Evaluating the current model
```shell
python3 get_accuracy.py
```

## Hosting the API
This section assumes an installation at /var/hateflow.
```shell
sudo mkdir /var/www/nltk_data
sudo mkdir /var/www/gensim-data
sudo chown www-data:www-data /var/www/*data
```
Apache has to be set up according to the system specific requirements.
### Apache configuration file
```
<VirtualHost *:80>
                ServerName example.org
                ServerAdmin admin@example.org

                WSGIDaemonProcess hateflow user=www-data group=www-data threads=5 home=/var/hateflow/
                WSGIScriptAlias / /var/hateflow/main.wsgi

                <Directory /var/hateflow/>
                        WSGIProcessGroup hateflow
                        WSGIApplicationGroup %{GLOBAL}
                        Require all granted
                </Directory>

                ErrorLog ${APACHE_LOG_DIR}/error.log
                LogLevel warn
                CustomLog ${APACHE_LOG_DIR}/access.log combined
</VirtualHost>
```

# Social Media Bots
```bash
cd hateflow/bots
pip3 install -r requirements.txt
```
## Twitch
1. Create an account on twitch.
2. Request an oauth code and update bots/.env accordingly.
3. Try running it: `python3 twitch.py`

This will start the bot in read mode. If you want to reply to detected comments, uncomment line 69 in `twitch.py`.


## Reddit
To use the reddit bot, just create a reddit account, create a reddit application and paste the credentials into the redditbot.py file. For a detailed description, follow the first few steps of this instructions:

https://yojji.io/blog/how-to-make-a-reddit-bot

Last but not least uncomment line 41 to atually reply to the detected comments. Now you can choose between scanning a whole Subreddit with many posts or just one post. Therefore call the function scan_subreddit or scan_comment.