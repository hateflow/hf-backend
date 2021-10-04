## Installation
(tested on Ubuntu 20.04 and Python 3.8)

```shell
cd /var
git clone git@github.com:jschoedl/neseps.git
sudo apt install python3-pip
cd neseps
pip3 install -r requirements.txt  --no-cache-dir
```

## Training a model
```shell
python3 nn_train.py
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
                WSGIScriptAlias / /var/neseps/main.wsgi
                WSGIApplicationGroup %{GLOBAL}
                <Directory /var/neseps/>
                        Require all granted
                </Directory>
                ErrorLog ${APACHE_LOG_DIR}/error.log
                LogLevel warn
                CustomLog ${APACHE_LOG_DIR}/access.log combined
</VirtualHost>
```
