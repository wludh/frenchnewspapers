def info(author, date)
  puts author + " kicked the bucket in " + date +"."
end

authors = {"Charles Dickens" => "1870", "William Thackeray" => "1863", "Anthony Trollope" => "1882", "Gerard Manley Hopkins" => "1889"}

authors.each do |author,date|
  info(author,date)
end