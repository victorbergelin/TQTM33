.PHONY	: all install clean tags dist userdoc devdoc
# DO NOT EDIT THIS FILE -- it was generated by configure.py
# configure.py --cuda-root=/usr/local/cuda/

all: tags
	/opt/local/Library/Frameworks/Python.framework/Versions/2.7/Resources/Python.app/Contents/MacOS/Python setup.py build

dist:
	/opt/local/Library/Frameworks/Python.framework/Versions/2.7/Resources/Python.app/Contents/MacOS/Python setup.py sdist

install: tags
	/opt/local/Library/Frameworks/Python.framework/Versions/2.7/Resources/Python.app/Contents/MacOS/Python setup.py install

clean:
	rm -Rf build
	rm -f tags

tags:
	ctags -R src || true

tests:
	echo "running tests"
	find ./test -type f -name "*.py" -exec python {} \;
