""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Simon Lindblad's vimrc
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" GENERAL CONFIG
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

" Setup syntax highlighting
syntax on
syntax enable

" Show alternatives in menu
set wildmenu

" Setup line numbers
set number
set relativenumber

" Highlight row and column 
set cursorline
set cursorcolumn

" Use dark solarized theme
set t_Co=256
set background=light
colorscheme bubblegum-256-light

" Incremental search. You do not need to type entire query.
set incsearch

" Always show statusline
set laststatus=2

" Configure indentation, replace tabs with 4 spaces.
set tabstop=4
set shiftwidth=4
set autoindent
set smartindent
set expandtab

" No backup
set nobackup
set nowritebackup

" Reload changes automatically
set autoread

" Ignore case when searching
set ignorecase

" Split windows shall appear on the right or below
set splitright
set splitbelow

" Use vim settings rather than vi
set nocompatible

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" GUI CONFIGURATION
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

" Remove menu bar
set guioptions-=m  

" Remove toolbar
set guioptions-=T  

" Remove right-hand scroll bar
set guioptions-=r  

" Remove left-hand scroll bar
set guioptions-=L  

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" PLUGINS
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

" Vundle stuff
filetype off
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
Plugin 'gmarik/Vundle.vim'

" File search
Plugin 'kien/ctrlp.vim'

Plugin 'NLKNguyen/papercolor-theme'
Plugin 'severin-lemaignan/vim-minimap.git'

" Style the statusline
Plugin 'bling/vim-airline'

" Syntax information
Plugin 'scrooloose/syntastic'

" Highlight next occurence of a character
Plugin 'unblevable/quick-scope'


" Manage snippets
Plugin 'MarcWeber/vim-addon-mw-utils'
Plugin 'tomtom/tlib_vim'
Plugin 'garbas/vim-snipmate'
Plugin 'honza/vim-snippets'

" Vundle stuff
call vundle#end()            
filetype plugin indent on    

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" PLUGIN CONFIGURATION 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

" Configure CtrlP Plugin
let g:ctrlp_map = '<c-p>'
let g:ctrlp_cmd = 'CtrlP'

" Configure syntastic
let g:syntastic_cpp_compiler = 'clang++'
let g:syntastic_cpp_compiler_options = ' -std=c++11 -stdlib=libc++'

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" CUSTOM VIM MAPPINGS
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

" Remapping navigation keys
imap ii <Esc>

" cd into the directory of the current file
command! CD cd %:p:h

" Shortcut to reload vimrc
map <leader>rr :source ~/.vimrc<CR>  

" Buffer shortcuts
nnoremap <leader><Tab>   :bnext<CR>
nnoremap <leader><S-Tab> :bprevious<CR>

" Use space to enter commands
nnoremap <Space> :

nnoremap <leader>c :set nohlsearch<CR>

" Force write
cmap w!! w !sudo tee % >/dev/null

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" LATEX 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

" Compile latex files
"nnoremap <leader>t :w<CR>:!rubber --pdf --warn all %<CR>
nnoremap <leader>t :w<CR>:!pdflatex %<CR>

" Function to calculate number of words in latex file.
function! WC()
	let filename = expand("%")
	let cmd = "detex " . filename . " | wc -w | tr -d [:space:]"
	let result = system(cmd)
	echo result . " words"
endfunction
command! WC call WC()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" WINDOW MANAGEMENT
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

" Shortcuts for navigating windows
map <leader>h <C-w>h
map <leader>j <C-w>j
map <leader>k <C-w>k
map <leader>l <C-w>l
map <leader>H <C-w>H
map <leader>J <C-w>J
map <leader>K <C-w>K
map <leader>L <C-w>L

" Shortcuts for splitting windows
map <leader>b <C-w>s<C-w>l
map <leader>v <C-w>v<C-w>j

set clipboard=unnamedplus
