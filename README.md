# HateFlow backend
HateFlow can detect and classify inappropriate comments. It is available using an API documented [here](http://docs.hateflow.de).
## Installation
(tested on Ubuntu 20.04 and Python 3.8)

```shell
cd /var
git clone git@github.com:jschoedl/hateflow.git
sudo apt install python3-pip
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

# Using the social media bots for twitch and reddit
## twitch
To use the twitch bot, first create an account on twitch, request an oauth code and register an app. For a detailed description, follow the first few steps of this instruction to get the bot running:

https://dev.to/ninjabunny9000/let-s-make-a-twitch-bot-with-python-2nd8

Last but not least uncomment line 69 in the bot.py file to atually reply to the detected comments.


## reddit
To use the reddit bot, just create a reddit account, create a reddit application and paste the credentials into the redditbot.py file. For a detailed description, follow the first few steps of this instructions:

https://yojji.io/blog/how-to-make-a-reddit-bot

Last but not least uncomment line 41 to atually reply to the detected comments. Now you can choose between scanning a whole Subreddit with many posts or just one post. Therefore call the function scan_subreddit or scan_comment.