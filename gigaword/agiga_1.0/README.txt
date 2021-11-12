Annotated Gigaword API and Command Line Tools v1.0 - July 21, 2012
------------------------------------------------------------------

This release includes a Java API and command line tools for reading
the Annotated Gigaword dataset XML files. 

-------------------
Project Hosting   :
-------------------

For the latest version, go to:
http://code.google.com/p/agiga

-------------------
Command Line Tools:
-------------------

The command line tools provide a convenient way to print human
readable versions of the XML annotations. The entry point is
edu.jhu.agiga.AgigaPrinter and it has the following usage.

usage: java edu.jhu.agiga.AgigaPrinter <type> <gzipped input file>
  where <type> is one of:
    words                     (Words only, one sentence per line)
    lemmas                    (Lemmas only, one sentence per line)
    pos                       (Part-of-speech tags)
    ner                       (Named entity types)
    basic-deps                (Basic dependency parses in CONNL-X format)
    col-deps                  (Collapsed dependency parses in CONNL-X format)
    col-ccproc-deps           (Collapsed and propagated dependency parses in CONNL-X format)
    phrase-structure          (Phrase structure parses)
    coref                     (Coreference resolution as SGML similar to MUC)
    stanford-deps             (toString() methods of Stanford dependency parse annotations)
    stanford-phrase-structure (toString() method of Stanford phrase structure parses)
    for-testing-only          (**For use in testing this API only**)
  and where <gzipped input file> is an .xml.gz file
  from Annotated Gigaword

For example, to print part-of-speech tags for the file
nyt_eng_199911.xml.gz, we could run:

java -cp build/agiga-1.0.jar:lib/* edu.jhu.agiga.AgigaPrinter pos annotated_gigaword/nyt_eng_199911.xml.gz

-------------------
Java API          :
-------------------

The Java API provides streaming access to the documents in the .xml.gz
files. Two iterators are provided: StreamingDocumentReader and
StreamingSentenceReader. Both of these take as input the path to an
Annotated Gigaword file and an AgigaPrefs object. 

By default, the AgigaPrefs constructor will ensure that every
annotation in the XML is read in and that the resulting objects are
fully populated. However, by turning off certain options, it's
possible to skip the reading and creation of objects corresponding to
unused annotations.

StreamingDocumentReader is an iterator over AgigaDocument objects. The
AgigaDocument class gives access to the coreference resolution (via
AgigaCoref objects) annotations and the sentences (via AgigaSentence
objects).

StreamingSentenceReader is an iterator over AgigaSentence
objects. This bypasses the document level annotations such as coref
and the document ids and provides direct access to the sentence
annotations only.

AgigaPrinter provides examples of how to use these iterators and set
the AgigaPrefs object so that only the necessary annotations are read.
Examples of how to use the Agiga objects can also be found in the
AgigaDocument.write* and AgigaSentence.write* methods.

----------------------
One- vs. Zero-Indexing:
----------------------

In the XML, the sentences and tokens are given Ids that are
one-indexed. However, we find it to be more convenient to work with
zero-indexed **indices** in the Java API. Accordingly, the Java API
does not provide access to these original Ids but instead provides
access to indices. These indices are accessed via methods named
get*Idx(), such as AgigaSentence.getIdx() and
AgigaMention.getSentenceIdx() -- or AgigaToken.getIdx() and
AgigaDependency.getGovIdx(). These indices also correspond to the
ordered elements in the Lists used throughout the API.

Of course, the original Ids from the XML can be recovered by adding
one to the indices in the API. However, we didn't want to confuse the
issue by providing API calls for both.

-------------------
Building          :
-------------------

A build.xml is provided for building with Apache Ant.  Example
commands are below and should be run from the top level directory that
contains the build.xml.

# To compile: 
ant

# To clean and compile
ant clean compile

# To build jars of classes and sources:
ant jar
